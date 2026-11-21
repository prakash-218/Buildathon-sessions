//! A minimal Parquet writer: PLAIN encoding, uncompressed, data page v1, no
//! statistics. Just enough for the two output tables, and much cheaper than a
//! general-purpose writer (no per-value allocation, one buffer per file).
//!
//! File layout: "PAR1", then per row group one data page per column, then the
//! Thrift-compact FileMetaData, its 4-byte length, and "PAR1".

/// Physical types (parquet.thrift `Type`).
const INT32: i32 = 1;
const INT64: i32 = 2;
const BYTE_ARRAY: i32 = 6;
/// `ConvertedType`.
const UTF8: i32 = 0;
const DATE: i32 = 6;
/// `Encoding`.
const PLAIN: i32 = 0;
const RLE: i32 = 3;

/// Rows per row group, so pages stay a few MB at most.
pub const GROUP_ROWS: usize = 1 << 17;

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Str,    // REQUIRED BYTE_ARRAY (STRING)
    OptStr, // OPTIONAL BYTE_ARRAY (STRING)
    Date,   // REQUIRED INT32 (DATE)
    I64,    // REQUIRED INT64
}

impl Kind {
    fn physical(self) -> i32 {
        match self {
            Kind::Str | Kind::OptStr => BYTE_ARRAY,
            Kind::Date => INT32,
            Kind::I64 => INT64,
        }
    }
}

/// One column of one row group, already PLAIN-encoded: `values` holds the
/// encoded values, and for `OptStr` `present` says which rows are non-null.
pub struct Column {
    pub values: Vec<u8>,
    pub present: Vec<bool>,
}

impl Column {
    pub fn new() -> Self {
        Column { values: Vec::new(), present: Vec::new() }
    }
    #[inline]
    pub fn i64(&mut self, v: i64) {
        self.values.extend_from_slice(&v.to_le_bytes());
    }
    #[inline]
    pub fn i32(&mut self, v: i32) {
        self.values.extend_from_slice(&v.to_le_bytes());
    }
    #[inline]
    pub fn bytes(&mut self, s: &[u8]) {
        self.values.extend_from_slice(&(s.len() as u32).to_le_bytes());
        self.values.extend_from_slice(s);
    }
    /// A string written by `f` straight into the buffer (no temporary).
    #[inline]
    pub fn bytes_with(&mut self, f: impl FnOnce(&mut Vec<u8>)) {
        let at = self.values.len();
        self.values.extend_from_slice(&[0; 4]);
        f(&mut self.values);
        let n = (self.values.len() - at - 4) as u32;
        self.values[at..at + 4].copy_from_slice(&n.to_le_bytes());
    }
    #[inline]
    pub fn opt_bytes(&mut self, s: Option<&[u8]>) {
        self.present.push(s.is_some());
        if let Some(s) = s {
            self.bytes(s);
        }
    }
}

/// Thrift compact protocol, only what FileMetaData and PageHeader need.
struct Thrift {
    b: Vec<u8>,
    last: Vec<i16>, // last field id of each open struct
}

const T_I32: u8 = 5;
const T_I64: u8 = 6;
const T_BINARY: u8 = 8;
const T_LIST: u8 = 9;
const T_STRUCT: u8 = 12;

impl Thrift {
    fn new() -> Self {
        Thrift { b: Vec::new(), last: vec![0] }
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.b.push(v as u8 | 0x80);
            v >>= 7;
        }
        self.b.push(v as u8);
    }
    fn zigzag(v: i64) -> u64 {
        ((v << 1) ^ (v >> 63)) as u64
    }
    fn field(&mut self, id: i16, ty: u8) {
        let last = self.last.last_mut().unwrap();
        let d = id - *last;
        *last = id;
        if (1..=15).contains(&d) {
            self.b.push((d as u8) << 4 | ty);
        } else {
            self.b.push(ty);
            self.varint(Self::zigzag(id as i64));
        }
    }
    fn i32(&mut self, id: i16, v: i32) {
        self.field(id, T_I32);
        self.varint(Self::zigzag(v as i64));
    }
    fn i64(&mut self, id: i16, v: i64) {
        self.field(id, T_I64);
        self.varint(Self::zigzag(v));
    }
    fn binary(&mut self, id: i16, s: &[u8]) {
        self.field(id, T_BINARY);
        self.varint(s.len() as u64);
        self.b.extend_from_slice(s);
    }
    fn list(&mut self, id: i16, elem: u8, n: usize) {
        self.field(id, T_LIST);
        if n < 15 {
            self.b.push((n as u8) << 4 | elem);
        } else {
            self.b.push(0xf0 | elem);
            self.varint(n as u64);
        }
    }
    /// A struct-typed field: begin, write its fields, then `end`.
    fn begin(&mut self, id: i16) {
        self.field(id, T_STRUCT);
        self.last.push(0);
    }
    /// A struct as a list element.
    fn begin_elem(&mut self) {
        self.last.push(0);
    }
    fn end(&mut self) {
        self.b.push(0); // STOP
        self.last.pop();
    }
    fn finish(mut self) -> Vec<u8> {
        self.b.push(0); // STOP of the outermost struct
        self.b
    }
}

fn page_header(num_values: usize, size: usize) -> Vec<u8> {
    let mut t = Thrift::new();
    t.i32(1, 0); // DATA_PAGE
    t.i32(2, size as i32); // uncompressed_page_size
    t.i32(3, size as i32); // compressed_page_size
    t.begin(5); // data_page_header
    t.i32(1, num_values as i32);
    t.i32(2, PLAIN);
    t.i32(3, RLE); // definition levels
    t.i32(4, RLE); // repetition levels
    t.end();
    t.finish()
}

/// Definition levels (bit width 1) as one RLE/bit-packed-hybrid run of
/// bit-packed groups of 8, preceded by its 4-byte length (data page v1).
fn def_levels(present: &[bool], out: &mut Vec<u8>) {
    let groups = present.len().div_ceil(8);
    let mut run = Vec::with_capacity(groups + 10);
    let mut h = (groups as u64) << 1 | 1; // bit-packed run header
    while h >= 0x80 {
        run.push(h as u8 | 0x80);
        h >>= 7;
    }
    run.push(h as u8);
    for g in present.chunks(8) {
        let mut byte = 0u8;
        for (i, &p) in g.iter().enumerate() {
            byte |= (p as u8) << i;
        }
        run.push(byte);
    }
    out.extend_from_slice(&(run.len() as u32).to_le_bytes());
    out.extend_from_slice(&run);
}

/// Write a Parquet file. `groups[g][c]` is column `c` of row group `g`, with
/// `rows[g]` rows; `schema` names each column and its kind.
pub fn write(path: &std::path::Path, schema: &[(&str, Kind)], groups: &[Vec<Column>], rows: &[usize]) -> std::io::Result<()> {
    let mut f: Vec<u8> = Vec::with_capacity(
        groups.iter().flatten().map(|c| c.values.len() + c.present.len() / 8 + 64).sum::<usize>() + 4096,
    );
    f.extend_from_slice(b"PAR1");
    // (data_page_offset, total size) per column per group
    let mut chunks: Vec<Vec<(i64, i64)>> = Vec::with_capacity(groups.len());
    let mut page = Vec::new();
    for (g, cols) in groups.iter().enumerate() {
        let mut meta = Vec::with_capacity(cols.len());
        for (c, col) in cols.iter().enumerate() {
            page.clear();
            if schema[c].1 == Kind::OptStr {
                def_levels(&col.present, &mut page);
            }
            let header = page_header(rows[g], page.len() + col.values.len());
            let offset = f.len() as i64;
            f.extend_from_slice(&header);
            f.extend_from_slice(&page);
            f.extend_from_slice(&col.values);
            meta.push((offset, f.len() as i64 - offset));
        }
        chunks.push(meta);
    }

    let mut t = Thrift::new();
    t.i32(1, 1); // version
    t.list(2, T_STRUCT, schema.len() + 1);
    t.begin_elem(); // root
    t.binary(4, b"schema");
    t.i32(5, schema.len() as i32); // num_children
    t.end();
    for &(name, kind) in schema {
        t.begin_elem();
        t.i32(1, kind.physical());
        t.i32(3, if kind == Kind::OptStr { 1 } else { 0 }); // OPTIONAL / REQUIRED
        t.binary(4, name.as_bytes());
        match kind {
            Kind::Str | Kind::OptStr => {
                t.i32(6, UTF8);
                t.begin(10); // logicalType
                t.begin(1); // STRING
                t.end();
                t.end();
            }
            Kind::Date => {
                t.i32(6, DATE);
                t.begin(10);
                t.begin(6); // DATE
                t.end();
                t.end();
            }
            Kind::I64 => {}
        }
        t.end();
    }
    t.i64(3, rows.iter().sum::<usize>() as i64);
    t.list(4, T_STRUCT, groups.len());
    for (g, meta) in chunks.iter().enumerate() {
        t.begin_elem();
        t.list(1, T_STRUCT, meta.len());
        for (c, &(offset, size)) in meta.iter().enumerate() {
            t.begin_elem(); // ColumnChunk
            t.i64(2, offset); // file_offset
            t.begin(3); // meta_data
            t.i32(1, schema[c].1.physical());
            t.list(2, T_I32, 2);
            t.varint(Thrift::zigzag(PLAIN as i64));
            t.varint(Thrift::zigzag(RLE as i64));
            t.list(3, T_BINARY, 1);
            t.varint(schema[c].0.len() as u64);
            t.b.extend_from_slice(schema[c].0.as_bytes());
            t.i32(4, 0); // UNCOMPRESSED
            t.i64(5, rows[g] as i64);
            t.i64(6, size);
            t.i64(7, size);
            t.i64(9, offset); // data_page_offset
            t.end();
            t.end();
        }
        t.i64(2, meta.iter().map(|m| m.1).sum()); // total_byte_size
        t.i64(3, rows[g] as i64);
        t.end();
    }
    t.binary(6, b"shopping-sessions pipeline");
    let footer = t.finish();
    f.extend_from_slice(&footer);
    f.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    f.extend_from_slice(b"PAR1");
    std::fs::write(path, f)
}

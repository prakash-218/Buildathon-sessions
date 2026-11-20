#!/usr/bin/env python3
"""
Shopping Sessions: slow, literal reference implementation of rules.md.

Purpose: a test oracle, not a contender. Every rule is written out plainly so it
can be checked against rules.md line by line. Speed does not matter here.

All IDs and values are kept as raw bytes, so comparisons are byte-wise and
nothing is ever normalized.

Usage: reference.py [DATA_DIR OUTPUT_DIR]   (or env DATA_DIR / OUTPUT_DIR)
"""
import bisect
import datetime as dt
import glob
import json
import os
import sys
from collections import defaultdict

import polars as pl

DAY_MS = 86_400_000
GAP_MS = 1_800_000
WEEK_MS = 7 * DAY_MS
LATE_MS = DAY_MS
EVENT_TYPES = {b"view", b"cart", b"purchase"}
DIGITS = frozenset(b"0123456789")
EPOCH = dt.date(1970, 1, 1)


# --------------------------------------------------------------------------- CSV


def parse_line(line, nfields):
    """One line (bytes, without the \n) -> list of values (None for empty), or None if malformed."""
    if not line:
        return None  # an empty line is malformed
    parts = line.split(b",")
    if len(parts) != nfields:
        return None
    out = []
    for p in parts:
        if b'"' in p:
            # Quotes are allowed only around a whole field, with none inside.
            if len(p) < 2 or p[0] != 0x22 or p[-1] != 0x22 or b'"' in p[1:-1]:
                return None
            p = p[1:-1]
        out.append(p if p else None)
    return out


def read_rows(path, nfields):
    """Yield parsed rows of one CSV file; the first line is the header and is skipped."""
    with open(path, "rb") as f:
        data = f.read()
    if not data:
        return
    lines = data.split(b"\n")
    if data.endswith(b"\n"):
        lines.pop()  # the final \n ends the last line; it does not start an empty one
    for line in lines[1:]:
        row = parse_line(line, nfields)
        if row is not None:
            yield row


def read_dir(data, sub, nfields):
    for path in sorted(glob.glob(os.path.join(data, sub, "*.csv"))):
        yield from read_rows(path, nfields)


# --------------------------------------------------------------------- Timestamps


def is_digits(b):
    return len(b) > 0 and all(c in DIGITS for c in b)


def days_from_civil(y, m, d):
    """Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm)."""
    y -= m <= 2
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def is_leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def parse_ts(s):
    """bytes -> epoch ms (int, may be huge for F3), or None if not one of F1/F2/F3."""
    if s is None:
        return None
    if is_digits(s):  # F3
        return int(s)
    n = len(s)
    if n not in (20, 25):
        return None
    # YYYY-MM-DDTHH:MM:SS
    if s[4] != 0x2D or s[7] != 0x2D or s[10] != 0x54 or s[13] != 0x3A or s[16] != 0x3A:
        return None
    for a, b in ((0, 4), (5, 7), (8, 10), (11, 13), (14, 16), (17, 19)):
        if not is_digits(s[a:b]):
            return None
    y, mo, d = int(s[0:4]), int(s[5:7]), int(s[8:10])
    h, mi, se = int(s[11:13]), int(s[14:16]), int(s[17:19])
    if not 1 <= mo <= 12:
        return None
    mdays = 29 if (mo == 2 and is_leap(y)) else MONTH_DAYS[mo - 1]
    if not 1 <= d <= mdays or h > 23 or mi > 59 or se > 59:
        return None
    if n == 20:
        if s[19] != 0x5A:  # 'Z' only, never 'z'
            return None
        off = 0
    else:
        sign = s[19]
        if sign not in (0x2B, 0x2D) or s[22] != 0x3A:
            return None
        if not is_digits(s[20:22]) or not is_digits(s[23:25]):
            return None
        oh, om = int(s[20:22]), int(s[23:25])
        if oh > 23 or om > 59:
            return None
        off = (oh * 60 + om) * 60_000
        if sign == 0x2D:
            off = -off
    ms = ((days_from_civil(y, mo, d) * 24 + h) * 60 + mi) * 60_000 + se * 1000
    return ms - off  # subtract the offset to get UTC


def iso_to_ms(text):
    v = parse_ts(text.encode())
    assert v is not None, text
    return v


# ------------------------------------------------------------------------ Links


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        p = self.parent
        root = x
        while p.get(root, root) != root:
            root = p[root]
        while x != root:  # path compression
            nxt = p.get(x, x)
            p[x] = root
            x = nxt
        return root

    def union(self, a, b):
        self.parent.setdefault(a, a)
        self.parent.setdefault(b, b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_person_map(data):
    uf = UnionFind()
    for a, b in read_dir(data, "links", 2):
        if a is None or b is None:
            continue
        uf.union(a, b)
    name = {}  # root -> smallest id
    for x in list(uf.parent):
        r = uf.find(x)
        if r not in name or x < name[r]:
            name[r] = x
    person = {x: name[uf.find(x)] for x in uf.parent}
    return lambda uid: person.get(uid, uid)


# ------------------------------------------------------------------------ Main


def main():
    data = sys.argv[1] if len(sys.argv) > 2 else os.environ.get("DATA_DIR", "/data")
    out = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("OUTPUT_DIR", "/output")

    with open(os.path.join(data, "manifest.json")) as f:
        manifest = json.load(f)
    ws = iso_to_ms(manifest["window_start"])
    we = iso_to_ms(manifest["window_end"])

    category = {r[0]: r[1] for r in read_dir_file(data, "products.csv", 2)}
    channel = {r[0]: r[1] for r in read_dir_file(data, "campaigns.csv", 2)}

    person_of = build_person_map(data)

    # ---- Events: parse, dedup (smallest ingest_ts) before validation.
    kept = {}  # event_id -> (ingest_ms, row)
    for row in read_dir(data, "events", 7):
        eid = row[0]
        if eid is None:
            continue
        ing = parse_ts(row[3])
        if ing is None:
            ing = float("inf")  # rules say ingest_ts is always valid F1; be safe
        cur = kept.get(eid)
        if cur is None or ing < cur[0]:
            kept[eid] = (ing, row)

    events_by_person = defaultdict(list)  # person -> [(ts, price_or_0, is_purchase, category)]
    for ing, row in kept.values():
        _, uid, ts_raw, _, etype, pid, price_raw = row
        if uid is None or pid is None or etype not in EVENT_TYPES:
            continue
        ts = parse_ts(ts_raw)
        if ts is None or not (ws <= ts < we):
            continue
        price = 0
        purchase = etype == b"purchase"
        if purchase:
            if price_raw is None or not is_digits(price_raw):
                continue
            price = int(price_raw)
            if not 1 <= price <= 10_000_000:
                continue
        if ing - ts > LATE_MS:
            continue
        events_by_person[person_of(uid)].append(
            (ts, price, purchase, category.get(pid, b"unknown"))
        )
    del kept

    # ---- Clicks: parse, dedup (any copy), validate.
    clicks = {}
    for row in read_dir(data, "clicks", 4):
        if row[0] is None:
            continue
        clicks.setdefault(row[0], row)
    clicks_by_person = defaultdict(list)  # person -> [(ts, click_id, campaign)]
    for cid, uid, camp, ts_raw in clicks.values():
        if uid is None or camp is None or camp not in channel:
            continue
        ts = parse_ts(ts_raw)
        if ts is None or not (ws - WEEK_MS <= ts < we):
            continue
        clicks_by_person[person_of(uid)].append((ts, cid, camp))
    del clicks
    for lst in clicks_by_person.values():
        lst.sort()

    # ---- Sessions, attribution, aggregation.
    user_daily = {}  # (person, day) -> [sessions, events, revenue, {cat: cents}]
    camp_daily = defaultdict(lambda: [0, 0, 0, 0])  # (campaign, day) -> [sess, purch_sess, rev, assisted]

    for person, evs in events_by_person.items():
        evs.sort(key=lambda e: e[0])
        sessions = []
        prev = None
        for e in evs:
            if prev is None or e[0] - prev > GAP_MS:
                sessions.append([])
            sessions[-1].append(e)
            prev = e[0]

        cl = clicks_by_person.get(person, [])
        cl_ts = [c[0] for c in cl]

        for sess in sessions:
            start = sess[0][0]
            day = start // DAY_MS
            n = len(sess)
            rev = sum(e[1] for e in sess if e[2])
            purchased = any(e[2] for e in sess)

            ud = user_daily.get((person, day))
            if ud is None:
                ud = user_daily[(person, day)] = [0, 0, 0, defaultdict(int)]
            ud[0] += 1
            ud[1] += n
            ud[2] += rev
            for e in sess:
                if e[2]:
                    ud[3][e[3]] += e[1]

            lo = bisect.bisect_left(cl_ts, start - WEEK_MS)
            hi = bisect.bisect_right(cl_ts, start)  # clicks with ts <= start
            if hi > lo:
                last_ts = cl_ts[hi - 1]
                first = bisect.bisect_left(cl_ts, last_ts, lo, hi)  # smallest click_id at that ts
                credit = cl[first][2]
                assists = {c[2] for c in cl[lo:hi]}
            else:
                credit = b"organic"
                assists = set()

            cd = camp_daily[(credit, day)]
            cd[0] += 1
            cd[1] += 1 if purchased else 0
            cd[2] += rev
            for camp in assists:
                camp_daily[(camp, day)][3] += 1

    # ---- Write outputs.
    def date_of(day):
        return EPOCH + dt.timedelta(days=day)

    def top(cats):
        if not cats:
            return None
        best = max(cats.values())
        return min(c for c, v in cats.items() if v == best).decode()

    ud_rows = {
        "user_id": [], "day": [], "sessions": [], "events": [], "revenue_cents": [], "top_category": [],
    }
    for (p, day), (s, n, rev, cats) in user_daily.items():
        ud_rows["user_id"].append(p.decode())
        ud_rows["day"].append(date_of(day))
        ud_rows["sessions"].append(s)
        ud_rows["events"].append(n)
        ud_rows["revenue_cents"].append(rev)
        ud_rows["top_category"].append(top(cats))

    cd_rows = {
        "campaign_id": [], "channel": [], "day": [], "sessions": [],
        "purchasing_sessions": [], "revenue_cents": [], "assisted_sessions": [],
    }
    for (camp, day), (s, ps, rev, asst) in camp_daily.items():
        cd_rows["campaign_id"].append(camp.decode())
        cd_rows["channel"].append("organic" if camp == b"organic" else channel[camp].decode())
        cd_rows["day"].append(date_of(day))
        cd_rows["sessions"].append(s)
        cd_rows["purchasing_sessions"].append(ps)
        cd_rows["revenue_cents"].append(rev)
        cd_rows["assisted_sessions"].append(asst)

    schemas = {
        "user_daily": {"user_id": pl.String, "day": pl.Date, "sessions": pl.Int64, "events": pl.Int64,
                       "revenue_cents": pl.Int64, "top_category": pl.String},
        "campaign_daily": {"campaign_id": pl.String, "channel": pl.String, "day": pl.Date, "sessions": pl.Int64,
                           "purchasing_sessions": pl.Int64, "revenue_cents": pl.Int64,
                           "assisted_sessions": pl.Int64},
    }
    for name, rows in (("user_daily", ud_rows), ("campaign_daily", cd_rows)):
        folder = os.path.join(out, name)
        os.makedirs(folder, exist_ok=True)
        pl.DataFrame(rows, schema=schemas[name]).write_parquet(os.path.join(folder, "part-0.parquet"))
    print(f"user_daily {len(ud_rows['user_id'])} rows, campaign_daily {len(cd_rows['campaign_id'])} rows",
          file=sys.stderr)


def read_dir_file(data, name, nfields):
    yield from read_rows(os.path.join(data, name), nfields)


if __name__ == "__main__":
    main()

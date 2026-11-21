# Shopping Sessions: Rust pipeline. The scorer builds this natively on linux/amd64.
FROM rust:1.85-bookworm AS build
WORKDIR /build
# x86-64-v3 (AVX2, BMI2) runs on the scorers (Cascade Lake or newer) and most dev
# machines. crt-static links glibc in, so the binary runs in an empty image.
# --target keeps these flags off the build scripts, which run on the build host.
ENV RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=+crt-static"
ARG TARGET=x86_64-unknown-linux-gnu

# Build the dependencies first so code changes do not rebuild them.
COPY pipeline/Cargo.toml pipeline/Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release --target $TARGET && rm -rf src

COPY pipeline/src ./src
RUN touch src/main.rs && cargo build --release --target $TARGET \
    && cp target/$TARGET/release/sessions /sessions

# An empty image with one static binary: nothing to load at start.
FROM scratch
COPY --from=build /sessions /sessions
# mimalloc: keep freed memory for reuse instead of returning it to the OS.
ENV MIMALLOC_PURGE_DELAY=-1
# The scorer runs as uid 1000 with a read-only root; the binary needs neither a home nor temp files.
USER 1000:1000
CMD ["/sessions"]

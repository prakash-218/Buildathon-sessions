# Shopping Sessions: Rust pipeline. The scorer builds this natively on linux/amd64.
FROM rust:1.85-bookworm AS build
WORKDIR /build
# The scorers are Cascade Lake or newer; x86-64-v3 (AVX2, BMI2) runs there and on most dev machines.
ENV RUSTFLAGS="-C target-cpu=x86-64-v3"

# Build the dependencies first so code changes do not rebuild them.
COPY pipeline/Cargo.toml pipeline/Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release && rm -rf src

COPY pipeline/src ./src
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim
COPY --from=build /build/target/release/sessions /usr/local/bin/sessions
# The scorer runs as uid 1000 with a read-only root; the binary needs neither a home nor temp files.
USER 1000:1000
CMD ["/usr/local/bin/sessions"]

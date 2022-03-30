FROM rust
WORKDIR /workdir
COPY . .
RUN cargo build --release
CMD ["./target/release/imagecli"]

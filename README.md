This is the code accompanying https://orlp.net/blog/taming-float-sums/.

To run the accuracy tests:

    cargo run --release
    
To run the benchmark:

    RUSTFLAGS="-C target-cpu=native" cargo bench
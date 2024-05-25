use std::time::Duration;

use rand::distributions::Uniform;
use rand::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

use sum_bench::*;

fn criterion_benchmark(c: &mut Criterion) {
    let n = 100_000;
    let v: Vec<f32> = thread_rng().sample_iter(Uniform::from(0.0..1.0)).take(n).collect();

    let mut g = c.benchmark_group("sums");
    g.throughput(criterion::Throughput::BytesDecimal(n as u64 * 4));
    g.warm_up_time(Duration::from_millis(200));
    g.measurement_time(Duration::from_secs(5));
    g.bench_function("naive", |b| b.iter(|| naive_sum(black_box(&v))));
    g.bench_function("naive_autovec", |b| b.iter(|| naive_sum_autovec(black_box(&v))));
    g.bench_function("pairwise", |b| b.iter(|| pairwise_sum(black_box(&v))));
    g.bench_function("block_pairwise", |b| b.iter(|| block_pairwise_sum(black_box(&v))));
    g.bench_function("block_pairwise_autovec", |b| b.iter(|| block_pairwise_sum_autovec(black_box(&v))));
    g.bench_function("kahan", |b| b.iter(|| kahan_sum(black_box(&v))));
    g.bench_function("block_kahan", |b| b.iter(|| block_kahan_sum(black_box(&v))));
    g.bench_function("block_kahan_autovec", |b| b.iter(|| block_kahan_sum_autovec(black_box(&v))));
    g.bench_function("crate_accurate_buffer", |b| b.iter(|| crate_accurate_buffer(black_box(&v))));
    g.bench_function("crate_accurate_inplace", |b| b.iter_batched_ref(|| v.to_owned(), |v| crate_accurate_inplace(black_box(v)), BatchSize::SmallInput));
    g.bench_function("crate_fsum", |b| b.iter(|| crate_fsum(black_box(&v))));
    g.bench_function("sum_orlp", |b| b.iter(|| sum_orlp(black_box(&v))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
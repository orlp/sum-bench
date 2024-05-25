use rand::prelude::*;
use rand::distributions::uniform::{UniformFloat, UniformSampler};
use rayon::prelude::*;

use sum_bench::*;


fn main() {
    let n = 100_000;
    let iters = 10_000;
    
    macro_rules! eval {
        ($f:ident) => {{
            let distr = UniformFloat::<f32>::new(-100_000.0, 100_000.0);
            let err: f64 = 
            (0..iters).into_par_iter().map(|i| {
                let mut rng = StdRng::seed_from_u64(0xeeadbeef ^ i);
                let mut v: Vec<f32> = (0..n).map(|_| distr.sample(&mut rng)).collect();
                let corr = crate_fsum(&v) as f64;
                ($f(&mut v) as f64 - corr).abs()
            }).sum();
            println!("{:<40} {:.10}", stringify!($f), err / iters as f64);
        }}
    }
    eval!(naive_sum);
    eval!(naive_sum_autovec);
    eval!(pairwise_sum);
    eval!(block_pairwise_sum);
    eval!(block_pairwise_sum_autovec);
    eval!(kahan_sum);
    eval!(block_kahan_sum);
    eval!(block_kahan_sum_autovec);
    eval!(crate_accurate_inplace);
    eval!(crate_accurate_buffer);
    eval!(crate_fsum);
    eval!(sum_orlp);
}

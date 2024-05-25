#![allow(internal_features)]
#![feature(core_intrinsics)]
use std::intrinsics::fadd_algebraic;

pub fn naive_sum(arr: &[f32]) -> f32 {
    let mut sum = 0.0;
    for x in arr {
        sum += *x;
    }
    sum
}

pub fn naive_sum_autovec(arr: &[f32]) -> f32 {
    let mut sum = 0.0;
    for x in arr {
        sum = fadd_algebraic(sum, *x);
    }
    sum
}

pub fn pairwise_sum(arr: &[f32]) -> f32 {
    if arr.len() == 0 { return 0.0; }
    if arr.len() == 1 { return arr[0]; }
    let (first, second) = arr.split_at(arr.len() / 2);
    pairwise_sum(first) + pairwise_sum(second)
}

pub fn block_pairwise_sum(arr: &[f32]) -> f32 {
    if arr.len() > 256 {
        let (first, second) = arr.split_at(arr.len() / 2);
        block_pairwise_sum(first) + block_pairwise_sum(second)
    } else {
        naive_sum(arr)
    }
}

pub fn block_pairwise_sum_autovec(arr: &[f32]) -> f32 {
    if arr.len() > 256 {
        let split = (arr.len() / 2).next_multiple_of(256);
        let (first, second) = arr.split_at(split);
        block_pairwise_sum_autovec(first) + block_pairwise_sum_autovec(second)
    } else {
        naive_sum_autovec(arr)
    }
}

pub fn kahan_sum(arr: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for x in arr {
        let y = *x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

pub fn block_kahan_sum(arr: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for chunk in arr.chunks(256) {
        let x = naive_sum(chunk);
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

pub fn block_kahan_sum_autovec(arr: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for chunk in arr.chunks(256) {
        let x = naive_sum_autovec(chunk);
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

pub fn crate_accurate_inplace(arr: &mut [f32]) -> f32 {
    accurate::sum::i_fast_sum_in_place(arr)
}

pub fn crate_accurate_buffer(arr: &[f32]) -> f32 {
    use accurate::traits::SumAccumulator;
    accurate::sum::OnlineExactSum::zero().absorb(arr.iter().copied()).sum()
}

pub fn crate_fsum(arr: &[f32]) -> f32 {
    fsum::FSum::with_all(arr.iter().map(|x| *x as f64)).value() as f32
}

fn sum_block(arr: &[f32]) -> f32 {
    arr.iter().fold(0.0, |x, y| fadd_algebraic(x, *y))
}

pub fn sum_orlp(arr: &[f32]) -> f32 {
    let mut chunks = arr.chunks_exact(256);
    let mut sum = 0.0;
    let mut c = 0.0;
    for chunk in &mut chunks {
        let y = sum_block(chunk) - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum + (sum_block(chunks.remainder()) - c)
}
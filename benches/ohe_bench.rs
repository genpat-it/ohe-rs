use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

// We benchmark the core functions directly since they're in the lib
// For now, benchmark the category discovery which is the main bottleneck

fn bench_category_discovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("category_discovery");

    for &(n, k) in &[
        (100_000, 10),
        (1_000_000, 10),
        (1_000_000, 1_000),
        (10_000_000, 10),
        (10_000_000, 1_000),
        (10_000_000, 100_000),
    ] {
        let data: Vec<i64> = (0..n).map(|i| (i % k) as i64).collect();

        group.bench_with_input(
            BenchmarkId::new("parallel", format!("N={n}_K={k}")),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(ohe_rs::discover_categories_parallel_pub(data));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_category_discovery);
criterion_main!(benches);

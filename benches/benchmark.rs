use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::{DynamicImage, ImageBuffer, Luma};
use imagecli::{
    image_ops::{parse, parse_pipeline},
    run_pipeline,
};

fn bench_pipeline_parsing(b: &mut Criterion) {
    let pipeline =
        black_box("gray > func { 255 * (p > 100) } > rotate 45 > othresh > scale 2 > resize w=7");
    b.bench_function("bench_pipeline_parsing", |b| {
        b.iter(|| {
            let pipeline = parse_pipeline(pipeline).unwrap();
            black_box(pipeline);
        })
    });
}

fn bench_run_pipeline_with_user_defined_func(b: &mut Criterion) {
    let pipeline = parse("func { 255 * (p > 100) }").unwrap();
    let image =
        DynamicImage::ImageLuma8(ImageBuffer::from_fn(100, 100, |x, y| Luma([(x + y) as u8])));
    b.bench_function("bench_run_pipeline_with_user_defined_func", |b| {
        b.iter(|| {
            let inputs = black_box(vec![image.clone()]);
            let _ = black_box(run_pipeline(&pipeline, inputs, false));
        })
    });
}

fn bench_run_pipeline(b: &mut Criterion) {
    let pipeline = parse("gray > DUP > rotate 45 > ROT 2 > othresh > hcat").unwrap();
    let image =
        DynamicImage::ImageLuma8(ImageBuffer::from_fn(100, 100, |x, y| Luma([(x + y) as u8])));
    b.bench_function("bench_run_pipeline", |b| {
        b.iter(|| {
            let inputs = black_box(vec![image.clone()]);
            let _ = black_box(run_pipeline(&pipeline, inputs, false));
        })
    });
}

criterion_group!(
    benches,
    bench_pipeline_parsing,
    bench_run_pipeline_with_user_defined_func,
    bench_run_pipeline
);
criterion_main!(benches);

use std::path::PathBuf;
use structopt::StructOpt;
use image::{DynamicImage, GenericImageView, open, imageops::FilterType};
use imageproc::filter::gaussian_blur_f32;

#[derive(StructOpt, Debug)]
struct Opt {
    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: u8,

    /// Input file
    #[structopt(short, long, parse(from_os_str))]
    input: PathBuf,

    /// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,

    /// Image processing pipeline to apply
    #[structopt(short, long)]
    pipeline: Option<String>,
}

// cargo run --release -- -v -i images/robin.jpg -o images/morphed.png -p 'gray > gaussian 10.0 > scale 0.3'
fn main() {
    let opt = Opt::from_args();
    let verbose = opt.verbose > 0;
    let input = open(&opt.input).unwrap();
    if verbose {
        println!("Input image {:?} (width: {}, height: {})", &opt.input, input.width(), input.height());
    }
    let output = if opt.pipeline.is_some() {
        let pipeline = parse_pipeline(&opt.pipeline.unwrap());
        process(input, &pipeline, opt.verbose > 0)
    } else {
        input
    };
    if verbose {
        println!("Output image {:?} (width: {}, height: {})", &opt.output, output.width(), output.height());
    }
    output.save(&opt.output).unwrap();
}

fn process(input: DynamicImage, pipeline: &[Stage], verbose: bool) -> DynamicImage {
    let mut output = input;
    for stage in pipeline {
        if verbose {
            println!("Applying stage {:?}", stage);
        }
        output = stage.apply(&output);
    }
    output
}

fn parse_pipeline(pipeline: &str) -> Vec<Stage> {
    pipeline.split('>').map(|s| s.trim()).map(parse_stage).collect()
}

fn parse_stage(stage: &str) -> Stage {
    let split: Vec<&str> = stage.split_whitespace().collect();
    match split[0] {
        "scale" => Stage::Scale(split[1].parse().unwrap()),
        "gaussian" => Stage::Gaussian(split[1].parse().unwrap()),
        "gray" => Stage::Gray,
        _ => panic!("Unrecognised stage name {}", split[0]),
    }
}

trait ImageExt {
    fn gaussian(&self, sigma: f32) -> Self;
}

impl ImageExt for DynamicImage {
    fn gaussian(&self, sigma: f32) -> Self {
        use DynamicImage::*;
        match self {
            ImageLuma8(image) => ImageLuma8(gaussian_blur_f32(image, sigma)),
            ImageLumaA8(image) => ImageLumaA8(gaussian_blur_f32(image, sigma)),
            ImageRgb8(image) => ImageRgb8(gaussian_blur_f32(image, sigma)),
            ImageRgba8(image) => ImageRgba8(gaussian_blur_f32(image, sigma)),
            ImageBgr8(image) => ImageBgr8(gaussian_blur_f32(image, sigma)),
            ImageBgra8(image) => ImageBgra8(gaussian_blur_f32(image, sigma)),
        }
    }
}

impl Stage {
    fn apply(&self, image: &DynamicImage) -> DynamicImage {
        match self {
            Self::Scale(s) => {
                let (w, h) = ((image.width() as f32 * s) as u32, (image.height() as f32 * s) as u32);
                image.resize(w, h, FilterType::Lanczos3)
            },
            Self::Gaussian(s) => {
                image.gaussian(*s)
            },
            Self::Gray => {
                image.grayscale()
            }
        }
    }
}

#[derive(Debug)]
enum Stage {
    /// Scale both width and height by given multiplier.
    Scale(f32),
    /// Apply a Gaussian blur with the given standard deviation.
    Gaussian(f32),
    /// Convert to grayscale.
    Gray
}

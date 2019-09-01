
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use imageproc::filter::gaussian_blur_f32;

#[derive(Debug)]
pub enum Stage {
    /// Scale both width and height by given multiplier.
    Scale(f32),
    /// Apply a Gaussian blur with the given standard deviation.
    Gaussian(f32),
    /// Convert to grayscale.
    Gray
}

impl Stage {
    pub fn apply(&self, image: &DynamicImage) -> DynamicImage {
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

    pub fn parse(stage: &str) -> Stage {
        let split: Vec<&str> = stage.split_whitespace().collect();
        match split[0] {
            "scale" => Stage::Scale(split[1].parse().unwrap()),
            "gaussian" => Stage::Gaussian(split[1].parse().unwrap()),
            "gray" => Stage::Gray,
            _ => panic!("Unrecognised stage name {}", split[0]),
        }
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


use image::{DynamicImage, GenericImageView, imageops::FilterType, Luma, LumaA, Rgb, Rgba, Bgr, Bgra};
use imageproc::{
    filter::gaussian_blur_f32,
    geometric_transformations::{Interpolation, rotate_about_center},
};

#[derive(Debug)]
pub enum Stage {
    /// Scale both width and height by given multiplier.
    Scale(f32),
    /// Apply a Gaussian blur with the given standard deviation.
    Gaussian(f32),
    /// Convert to grayscale.
    Gray,
    /// Rotate clockwise about the image's center by the given angle in degrees.
    Rotate(f32)
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
            },
            Self::Rotate(t) => {
                image.rotate(*t)
            }
        }
    }

    pub fn parse(stage: &str) -> Stage {
        let split: Vec<&str> = stage.split_whitespace().collect();
        match split[0] {
            "scale" => Stage::Scale(split[1].parse().unwrap()),
            "gaussian" => Stage::Gaussian(split[1].parse().unwrap()),
            "gray" => Stage::Gray,
            "rotate" => Stage::Rotate(split[1].parse().unwrap()),
            _ => panic!("Unrecognised stage name {}", split[0]),
        }
    }
}

trait ImageExt {
    fn gaussian(&self, sigma: f32) -> Self;
    fn rotate(&self, theta: f32) -> Self;
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

    fn rotate(&self, theta: f32) -> Self {
        use DynamicImage::*;
        let rad = theta * std::f32::consts::PI / 180.0;
        match self {
            ImageLuma8(image) => ImageLuma8(
                rotate_about_center(image, rad, Interpolation::Bilinear, Luma([0]))
            ),
            ImageLumaA8(image) => ImageLumaA8(
                rotate_about_center(image, rad, Interpolation::Bilinear, LumaA([0, 0]))
            ),
            ImageRgb8(image) => ImageRgb8(
                rotate_about_center(image, rad, Interpolation::Bilinear, Rgb([0, 0, 0]))
            ),
            ImageRgba8(image) => ImageRgba8(
                rotate_about_center(image, rad, Interpolation::Bilinear, Rgba([0, 0, 0, 0]))
            ),
            ImageBgr8(image) => ImageBgr8(
                rotate_about_center(image, rad, Interpolation::Bilinear, Bgr([0, 0, 0]))
            ),
            ImageBgra8(image) => ImageBgra8(
                rotate_about_center(image, rad, Interpolation::Bilinear, Bgra([0, 0, 0, 0]))
            ),
        }
    }
}

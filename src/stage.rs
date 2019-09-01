
use image::{
    DynamicImage, DynamicImage::*,
    GenericImageView, imageops::FilterType, Luma, LumaA, Rgb, Rgba, Bgr, Bgra
};
use imageproc::{
    contrast::{adaptive_threshold, otsu_level, threshold},
    definitions::Clamp,
    filter::gaussian_blur_f32,
    geometric_transformations::{Interpolation, rotate_about_center},
    gradients::sobel_gradient_map,
    seam_carving::shrink_width,
};
use std::cmp;

#[derive(Debug)]
pub enum Stage {
    /// Scale both width and height by given multiplier.
    Scale(f32),
    /// Apply a Gaussian blur with the given standard deviation.
    Gaussian(f32),
    /// Convert to grayscale.
    Gray,
    /// Rotate clockwise about the image's center by the given angle in degrees.
    Rotate(f32),
    /// Compute image gradients using the Sobel filter.
    Sobel,
    /// Apply seam carving to shrink the image to a provided multiple of its original width.
    Carve(f32),
    /// Binarises the image using an adaptive thresholding with given block radius.
    AdaptiveThreshold(u32),
    /// Binarises the image using Otsu thresholding.
    OtsuThreshold,
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
            },
            Self::Sobel => {
                image.sobel()
            },
            Self::Carve(r) => {
                image.carve(*r)
            },
            Self::AdaptiveThreshold(r) => {
                image.adaptive_threshold(*r)
            },
            Self::OtsuThreshold => {
                image.otsu_threshold()
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
            "sobel" => Stage::Sobel,
            "carve" => Stage::Carve(split[1].parse().unwrap()),
            "athresh" => Stage::AdaptiveThreshold(split[1].parse().unwrap()),
            "othresh" => Stage::OtsuThreshold,
            _ => panic!("Unrecognised stage name {}", split[0]),
        }
    }
}

trait ImageExt {
    fn gaussian(&self, sigma: f32) -> Self;
    fn rotate(&self, theta: f32) -> Self;
    fn sobel(&self) -> Self;
    fn carve(&self, ratio: f32) -> Self;
    fn adaptive_threshold(&self, block_radius: u32) -> Self;
    fn otsu_threshold(&self) -> Self;
}

impl ImageExt for DynamicImage {
    fn gaussian(&self, sigma: f32) -> Self {
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

    fn sobel(&self) -> Self {
        let clamp_to_u8 = |x| { <u8 as Clamp<u16>>::clamp(x) };
        match self {
            ImageLuma8(image) => ImageLuma8(
                sobel_gradient_map(image, |p| Luma([clamp_to_u8(p[0])]))
            ),
            ImageLumaA8(image) => ImageLuma8(
                sobel_gradient_map(image, |p| Luma([clamp_to_u8(p[0])]))
            ),
            ImageRgb8(image) => ImageLuma8(
                sobel_gradient_map(image, |p| Luma([clamp_to_u8(cmp::max(cmp::max(p[0], p[1]), p[2]))]))
            ),
            ImageRgba8(image) => ImageLuma8(
                sobel_gradient_map(image, |p| Luma([clamp_to_u8(cmp::max(cmp::max(p[0], p[1]), p[2]))]))
            ),
            ImageBgr8(image) => ImageLuma8(
                sobel_gradient_map(image, |p| Luma([clamp_to_u8(cmp::max(cmp::max(p[0], p[1]), p[2]))]))
            ),
            ImageBgra8(image) => ImageLuma8(
                sobel_gradient_map(image, |p| Luma([clamp_to_u8(cmp::max(cmp::max(p[0], p[1]), p[2]))]))
            ),
        }
    }

    fn carve(&self, ratio: f32) -> Self {
        assert!(ratio <= 1.0);
        let target_width = (self.width() as f32 * ratio) as u32;
        match self {
            ImageLuma8(image) => ImageLuma8(shrink_width(image, target_width)),
            ImageLumaA8(image) => ImageLumaA8(shrink_width(image, target_width)),
            ImageRgb8(image) => ImageRgb8(shrink_width(image, target_width)),
            ImageRgba8(image) => ImageRgba8(shrink_width(image, target_width)),
            ImageBgr8(image) => ImageBgr8(shrink_width(image, target_width)),
            ImageBgra8(image) => ImageBgra8(shrink_width(image, target_width)),
        }
    }

    fn adaptive_threshold(&self, block_radius: u32) -> Self {
        let gray = self.to_luma();
        ImageLuma8(adaptive_threshold(&gray, block_radius))
    }

    fn otsu_threshold(&self) -> Self {
        let gray = self.to_luma();
        ImageLuma8(threshold(&gray, otsu_level(&gray)))
    }
}


use image::{
    DynamicImage, DynamicImage::*,
    GenericImage, GenericImageView, imageops::FilterType, Luma, LumaA, Rgb, Rgba, Bgr, Bgra
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
use std::collections::HashMap;

/// A variable in the image pipeline.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Var(String);

pub struct Graph {
    pipelines: Vec<(Var, Vec<Stage>)>
}

impl Graph {
    pub fn parse(definition: &str) -> Self {
        let mut pipelines = Vec::new();
        for p in definition.split(';') {
            // TODO: only the final pipeline is allowed to be missing a variable
            // TODO: topological sort?
            // TODO: validate input (e.g. only one equals sign). Create a proper grammar
            let split: Vec<&str> = p.split('=').map(|s| s.trim()).collect();
            let name = if split.len() > 1 { Var(split[0].into()) } else { Var(String::from("FINAL")) };
            let stages = split.last().unwrap().split('>').map(|s| s.trim()).map(Stage::parse).collect();
            pipelines.push((name, stages));
        }
        Graph { pipelines }
    }

    pub fn process(&self, input: DynamicImage, verbose: bool) -> DynamicImage {
        let mut var_to_image = HashMap::new();
        for (var, pipeline) in &self.pipelines {
            let mut output = input.clone();
            // TODO: write something that makes a bit more sense!
            for stage in pipeline {
                if verbose {
                    println!("Applying stage {:?}", stage);
                }
                output = match stage {
                    Stage::HCat(l, r) => {
                        let inputs = vec![&var_to_image[&l], &var_to_image[&r]];
                        stage.apply(&inputs)
                    },
                    _ => {
                        let inputs = vec![&output];
                        stage.apply(&inputs)
                    }
                }
            }
            var_to_image.insert(var, output);
        }
        var_to_image[&Var(String::from("FINAL"))].clone()
    }
}

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
    /// The horizontal concatenation of two named images.
    HCat(Var, Var),
}

impl Stage {
    pub fn apply(&self, images: &[&DynamicImage]) -> DynamicImage {
        match self {
            Self::Scale(s) => {
                let image = images[0];
                let (w, h) = ((image.width() as f32 * s) as u32, (image.height() as f32 * s) as u32);
                image.resize(w, h, FilterType::Lanczos3)
            },
            Self::Gaussian(s) => {
                let image = images[0];
                image.gaussian(*s)
            },
            Self::Gray => {
                let image = images[0];
                image.grayscale()
            },
            Self::Rotate(t) => {
                let image = images[0];
                image.rotate(*t)
            },
            Self::Sobel => {
                let image = images[0];
                image.sobel()
            },
            Self::Carve(r) => {
                let image = images[0];
                image.carve(*r)
            },
            Self::AdaptiveThreshold(r) => {
                let image = images[0];
                image.adaptive_threshold(*r)
            },
            Self::OtsuThreshold => {
                let image = images[0];
                image.otsu_threshold()
            },
            Self::HCat(_l, _r) => {
                hcat(images[0], images[1])
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
            "hcat" => Stage::HCat(Var(split[1].into()), Var(split[2].into())),
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

fn hcat(left: &DynamicImage, right: &DynamicImage) -> DynamicImage {
    // TODO: do this properly!
    let left = left.to_luma();
    let right = right.to_luma();
    let (w, h) = (left.width() + right.width(), cmp::max(left.height(), right.height()));
    let mut out = image::ImageBuffer::new(w, h);
    out.copy_from(&left, 0, 0);
    out.copy_from(&right, left.width(), 0);
    ImageLuma8(out)
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

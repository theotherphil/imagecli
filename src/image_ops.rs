
use image::{
    DynamicImage, DynamicImage::*, GenericImage, GenericImageView,
    Luma, LumaA, Rgb, Rgba, Bgr, Bgra
};
use imageproc::definitions::Clamp;
use std::cmp;

use crate::ImageStack;

/// Image processing operations which read from and write to the image stack.
#[derive(Debug)]
pub enum ImageOp {
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
    /// Binarise the image using an adaptive thresholding with given block radius.
    AdaptiveThreshold(u32),
    /// Binarise the image using Otsu thresholding.
    OtsuThreshold,
    /// Horizontally concatenate two images.
    HCat,
    /// Vertically concatenate two images.
    VCat,
    /// Arrange images into a grid. First argument is the number of columns and second the number of rows.
    Grid(u32, u32),
    /// Apply a median filter with the given x radius and y radius.
    Median(u32, u32),
}

impl ImageOp {
    pub fn apply(&self, stack: &mut ImageStack) {
        let result = match self {
            Self::Scale(s) => {
                let image = stack.pop();
                scale(&image, *s)
            },
            Self::Gaussian(s) => {
                let image = stack.pop();
                gaussian(&image, *s)
            },
            Self::Gray => {
                let image = stack.pop();
                image.grayscale()
            },
            Self::Rotate(t) => {
                let image = stack.pop();
                rotate(&image, *t)
            },
            Self::Sobel => {
                let image = stack.pop();
                sobel(&image)
            },
            Self::Carve(r) => {
                let image = stack.pop();
                carve(&image, *r)
            },
            Self::AdaptiveThreshold(r) => {
                let image = stack.pop();
                adaptive_threshold(&image, *r)
            },
            Self::OtsuThreshold => {
                let image = stack.pop();
                otsu_threshold(&image)
            },
            Self::HCat => {
                let l = stack.pop();
                let r = stack.pop();
                hcat(&l, &r)
            },
            Self::VCat => {
                let l = stack.pop();
                let r = stack.pop();
                vcat(&l, &r)
            },
            Self::Grid(cols, rows) => {
                let t = stack.pop_n(*cols as usize * *rows as usize);
                grid(*cols, *rows, t)
            },
            Self::Median(x_radius, y_radius) => {
                let image = stack.pop();
                median(&image, *x_radius, *y_radius)
            }
        };
        stack.push(result);
    }

    pub fn parse(op: &str) -> Option<ImageOp> {
        let split: Vec<&str> = op.split_whitespace().collect();
        match split[0] {
            "scale" => Some(ImageOp::Scale(split[1].parse().unwrap())),
            "gaussian" => Some(ImageOp::Gaussian(split[1].parse().unwrap())),
            "gray" => Some(ImageOp::Gray),
            "rotate" => Some(ImageOp::Rotate(split[1].parse().unwrap())),
            "sobel" => Some(ImageOp::Sobel),
            "carve" => Some(ImageOp::Carve(split[1].parse().unwrap())),
            "athresh" => Some(ImageOp::AdaptiveThreshold(split[1].parse().unwrap())),
            "othresh" => Some(ImageOp::OtsuThreshold),
            "hcat" => Some(ImageOp::HCat),
            "vcat" => Some(ImageOp::VCat),
            "grid" => Some(ImageOp::Grid(split[1].parse().unwrap(), split[2].parse().unwrap())),
            "median" => Some(ImageOp::Median(split[1].parse().unwrap(), split[2].parse().unwrap())),
            _ => None,
        }
    }
}

fn hcat(left: &DynamicImage, right: &DynamicImage) -> DynamicImage {
    // TODO: handle formats properly
    let left = left.to_rgba();
    let right = right.to_rgba();
    let (w, h) = (left.width() + right.width(), cmp::max(left.height(), right.height()));
    let mut out = image::ImageBuffer::new(w, h);
    out.copy_from(&left, 0, 0);
    out.copy_from(&right, left.width(), 0);
    ImageRgba8(out)
}

fn vcat(top: &DynamicImage, bottom: &DynamicImage) -> DynamicImage {
    // TODO: handle formats properly
    let top = top.to_rgba();
    let bottom = bottom.to_rgba();
    let (w, h) = (cmp::max(top.width(), bottom.width()), top.height() + bottom.height());
    let mut out = image::ImageBuffer::new(w, h);
    out.copy_from(&top, 0, 0);
    out.copy_from(&bottom, 0, top.height());
    ImageRgba8(out)
}

fn grid(_cols: u32, _rows: u32, images: Vec<DynamicImage>) -> DynamicImage {
    // TODO: actually implement grid!
    let top = hcat(&images[0], &images[1]);
    let bottom = hcat(&images[2], &images[3]);
    vcat(&top, &bottom)
}

fn median(image: &DynamicImage, x_radius: u32, y_radius: u32) -> DynamicImage {
    use imageproc::filter::median_filter;
    match image {
        ImageLuma8(image) => ImageLuma8(median_filter(image, x_radius, y_radius)),
        ImageLumaA8(image) => ImageLumaA8(median_filter(image, x_radius, y_radius)),
        ImageRgb8(image) => ImageRgb8(median_filter(image, x_radius, y_radius)),
        ImageRgba8(image) => ImageRgba8(median_filter(image, x_radius, y_radius)),
        ImageBgr8(image) => ImageBgr8(median_filter(image, x_radius, y_radius)),
        ImageBgra8(image) => ImageBgra8(median_filter(image, x_radius, y_radius)),
    }
}

fn scale(image: &DynamicImage, scale: f32) -> DynamicImage {
    let (w, h) = ((image.width() as f32 * scale) as u32, (image.height() as f32 * scale) as u32);
    image.resize(w, h, image::imageops::FilterType::Lanczos3)
}

fn gaussian(image: &DynamicImage, sigma: f32) -> DynamicImage {
    use imageproc::filter::gaussian_blur_f32;
    match image {
        ImageLuma8(image) => ImageLuma8(gaussian_blur_f32(image, sigma)),
        ImageLumaA8(image) => ImageLumaA8(gaussian_blur_f32(image, sigma)),
        ImageRgb8(image) => ImageRgb8(gaussian_blur_f32(image, sigma)),
        ImageRgba8(image) => ImageRgba8(gaussian_blur_f32(image, sigma)),
        ImageBgr8(image) => ImageBgr8(gaussian_blur_f32(image, sigma)),
        ImageBgra8(image) => ImageBgra8(gaussian_blur_f32(image, sigma)),
    }
}

fn rotate(image: &DynamicImage, theta: f32) -> DynamicImage {
    use imageproc::geometric_transformations::{Interpolation, rotate_about_center};
    let rad = theta * std::f32::consts::PI / 180.0;
    match image {
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

fn sobel(image: &DynamicImage) -> DynamicImage {
    use imageproc::gradients::sobel_gradient_map;
    let clamp_to_u8 = |x| { <u8 as Clamp<u16>>::clamp(x) };
    match image {
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

fn carve(image: &DynamicImage, ratio: f32) -> DynamicImage {
    use imageproc::seam_carving::shrink_width;
    assert!(ratio <= 1.0);
    let target_width = (image.width() as f32 * ratio) as u32;
    match image {
        ImageLuma8(image) => ImageLuma8(shrink_width(image, target_width)),
        ImageLumaA8(image) => ImageLumaA8(shrink_width(image, target_width)),
        ImageRgb8(image) => ImageRgb8(shrink_width(image, target_width)),
        ImageRgba8(image) => ImageRgba8(shrink_width(image, target_width)),
        ImageBgr8(image) => ImageBgr8(shrink_width(image, target_width)),
        ImageBgra8(image) => ImageBgra8(shrink_width(image, target_width)),
    }
}

fn adaptive_threshold(image: &DynamicImage, block_radius: u32) -> DynamicImage {
    let gray = image.to_luma();
    ImageLuma8(imageproc::contrast::adaptive_threshold(&gray, block_radius))
}

fn otsu_threshold(image: &DynamicImage) -> DynamicImage {
    use imageproc::contrast::{otsu_level, threshold};
    let gray = image.to_luma();
    ImageLuma8(threshold(&gray, otsu_level(&gray)))
}

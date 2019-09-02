
use image::{
    DynamicImage, DynamicImage::*, GenericImage, GenericImageView,
    Luma, LumaA, Rgb, Rgba, Bgr, Bgra
};
use imageproc::definitions::Clamp;
use std::cmp;

use crate::ImageStack;

/// An image processing operation that operates on a stack of images.
pub trait ImageOp : std::fmt::Debug {
    fn apply(&self, stack: &mut ImageStack);
}

pub fn parse(op: &str) -> Option<Box<dyn ImageOp>> {
    if op.starts_with("[") && op.ends_with("]") {
        // TODO: find the method I actually want here when I have internet again
        let ch: Vec<char> = op.chars().skip(1).collect();
        let ch = &ch[..ch.len() - 2];
        let op: String = ch.iter().collect();
        let split: Vec<&str> = op.split(',').map(|s| s.trim()).collect();
        let mut ops: Vec<Box<dyn ImageOp>> = Vec::new();
        for s in split {
            if let Some(o) = parse(s) {
                ops.push(o);
            } else {
                return None;
            }
        }
        return Some(Box::new(Array(ops)));
    }
    let split: Vec<&str> = op.split_whitespace().collect();
    match split[0] {
        "scale" => Some(Box::new(Scale(split[1].parse().unwrap()))),
        "gaussian" => Some(Box::new(Gaussian(split[1].parse().unwrap()))),
        "gray" => Some(Box::new(Gray)),
        "rotate" => Some(Box::new(Rotate(split[1].parse().unwrap()))),
        "sobel" => Some(Box::new(Sobel)),
        "carve" => Some(Box::new(Carve(split[1].parse().unwrap()))),
        "athresh" => Some(Box::new(AdaptiveThreshold(split[1].parse().unwrap()))),
        "othresh" => Some(Box::new(OtsuThreshold)),
        "hcat" => Some(Box::new(HCat)),
        "vcat" => Some(Box::new(VCat)),
        "grid" => Some(Box::new(Grid(split[1].parse().unwrap(), split[2].parse().unwrap()))),
        "median" => Some(Box::new(Median(split[1].parse().unwrap(), split[2].parse().unwrap()))),
        "red" => Some(Box::new(Red)),
        "green" => Some(Box::new(Green)),
        "blue" => Some(Box::new(Blue)),
        _ => None,
    }
}

// TODO: use macros here and generate the whole impl
fn one_in_one_out<F>(stack: &mut ImageStack, f: F)
where
    F: FnOnce(&DynamicImage) -> DynamicImage
{
    let image = stack.pop();
    let result = f(&image);
    stack.push(result);    
}

fn two_in_one_out<F>(stack: &mut ImageStack, f: F)
where
    F: FnOnce(&DynamicImage, &DynamicImage) -> DynamicImage
{
    let left = stack.pop();
    let right = stack.pop();
    let result = f(&left, &right);
    stack.push(result);
}

/// Scale both width and height by given multiplier.
#[derive(Debug)]
struct Scale(f32);

impl ImageOp for Scale {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| scale(i, self.0));
    }
}

fn scale(image: &DynamicImage, scale: f32) -> DynamicImage {
    let (w, h) = ((image.width() as f32 * scale) as u32, (image.height() as f32 * scale) as u32);
    image.resize(w, h, image::imageops::FilterType::Lanczos3)
}

/// Apply a Gaussian blur with the given standard deviation.
#[derive(Debug)]
struct Gaussian(f32);

impl ImageOp for Gaussian {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| gaussian(i, self.0));
    }
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

/// Convert to grayscale.
#[derive(Debug)]
struct Gray;

impl ImageOp for Gray {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| i.grayscale());
    }
}

 /// Rotate clockwise about the image's center by the given angle in degrees.
#[derive(Debug)]
struct Rotate(f32);

impl ImageOp for Rotate {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| rotate(i, self.0));
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

/// Compute image gradients using the Sobel filter.
#[derive(Debug)]
struct Sobel;

impl ImageOp for Sobel {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| sobel(i));
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

/// Apply seam carving to shrink the image to a provided multiple of its original width.
#[derive(Debug)]
struct Carve(f32);

impl ImageOp for Carve {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| carve(i, self.0));
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

/// Binarise the image using an adaptive thresholding with given block radius.
#[derive(Debug)]
struct AdaptiveThreshold(u32);

impl ImageOp for AdaptiveThreshold {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| adaptive_threshold(i, self.0));
    }
}

fn adaptive_threshold(image: &DynamicImage, block_radius: u32) -> DynamicImage {
    let gray = image.to_luma();
    ImageLuma8(imageproc::contrast::adaptive_threshold(&gray, block_radius))
}

/// Binarise the image using Otsu thresholding.
#[derive(Debug)]
struct OtsuThreshold;

impl ImageOp for OtsuThreshold {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, otsu_threshold);
    }
}

fn otsu_threshold(image: &DynamicImage) -> DynamicImage {
    use imageproc::contrast::{otsu_level, threshold};
    let gray = image.to_luma();
    ImageLuma8(threshold(&gray, otsu_level(&gray)))
}

/// Horizontally concatenate two images.
#[derive(Debug)]
struct HCat;

impl ImageOp for HCat {
    fn apply(&self, stack: &mut ImageStack) {
        two_in_one_out(stack, |i, j| hcat(i, j));
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

/// Vertically concatenate two images.
#[derive(Debug)]
struct VCat;

impl ImageOp for VCat {
    fn apply(&self, stack: &mut ImageStack) {
        two_in_one_out(stack, |i, j| vcat(i, j));
    }
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

/// Arrange images into a grid. First argument is the number of columns and second the number of rows.
#[derive(Debug)]
struct Grid(u32, u32);

impl ImageOp for Grid {
    fn apply(&self, stack: &mut ImageStack) {
        // TODO: actually implement grid!
        let images = stack.pop_n(4);
        let top = hcat(&images[0], &images[1]);
        let bottom = hcat(&images[2], &images[3]);
        stack.push(vcat(&top, &bottom));
    }
}

/// Apply a median filter with the given x radius and y radius.
#[derive(Debug)]
struct Median(u32, u32);

impl ImageOp for Median {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| median(i, self.0, self.1));
    }
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

/// Applies the nth image operation to the nth element of the stack.
/// TODO: handle image operations other than one-in-one-out.
#[derive(Debug)]
struct Array(Vec<Box<dyn ImageOp>>);

impl ImageOp for Array {
    fn apply(&self, stack: &mut ImageStack) {
        let mut results = Vec::new();
        for op in &self.0 {
            // TODO: something less egregious
            op.apply(stack);
            results.push(stack.pop());
        }
        for result in results.into_iter().rev() {
            stack.push(result);
        }
    }
}

/// Extract the red channel from an image.
#[derive(Debug)]
struct Red;

impl ImageOp for Red {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, red);
    }
}

fn red(image: &DynamicImage) -> DynamicImage {
    use imageproc::map::red_channel;
    let rgb = image.to_rgb();
    ImageLuma8(red_channel(&rgb))
}

/// Extract the green channel from an image.
#[derive(Debug)]
struct Green;

impl ImageOp for Green {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, green);
    }
}

fn green(image: &DynamicImage) -> DynamicImage {
    use imageproc::map::green_channel;
    let rgb = image.to_rgb();
    ImageLuma8(green_channel(&rgb))
}

/// Extract the blue channel from an image.
#[derive(Debug)]
struct Blue;

impl ImageOp for Blue {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, blue);
    }
}

fn blue(image: &DynamicImage) -> DynamicImage {
    use imageproc::map::blue_channel;
    let rgb = image.to_rgb();
    ImageLuma8(blue_channel(&rgb))
}
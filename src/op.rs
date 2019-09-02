
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
use crate::stack::Stack;
use std::cmp;

// TODO:
// - hcat should support an arbitrary number of arguments - either read everything left on the stack or take num_imgs argument
// - at the end of processing, read images off the stack and write as many as we have output names for
//      example flows
//          read two inputs
//              first: blur
//              second: sharpen
//          finally: hcat blur sharpen
// 
//          read one input
//              hcat (red channel input) (blue channel input)
// 
//          read one input 
//              write red, blue and green channels as separate images
//  - add vcat
//  - add constant images
//  - add ability to add margins
//  - support user-defined functions over pixels, e.g. map \x -> x * 0.3
//  - add support for everything else relevant from imageproc
//  - add validation for parsing of individual stages (make them subcommands somehow?)
//  - add specific subcommands for the pipeline stages?
//      e.g. make -p 'scale 0.4' equivalent to --scale 0.4
//  - add stages with optional parameters. e.g. rotate uses image centre by default, but user
//      can specify a centre of rotation if they like

type ImageStack = Stack<DynamicImage>;

// TODO: support returning multiple images
pub fn run_pipeline(pipeline: &str, inputs: Vec<DynamicImage>, verbose: bool) -> DynamicImage {
    // TODO: validation!
    let ops: Vec<Op> = pipeline.split('>').map(|s| s.trim()).map(|s| Op::parse(s).unwrap()).collect();
    let mut stack = ImageStack::new(inputs);

    for op in ops {
        if verbose {
            println!("Applying {:?}", op);
        }
        op.apply(&mut stack);
    }

    stack.pop()
}

#[derive(Debug)]
pub enum Op {
    StackOp(StackOp),
    ImageOp(ImageOp),
}

impl Op {
    fn parse(op: &str) -> Option<Op> {
        match StackOp::parse(op) {
            Some(o) => Some(Op::StackOp(o)),
            None => match ImageOp::parse(op) {
                Some(o) => Some(Op::ImageOp(o)),
                None => None,
            }
        }
    }

    fn apply(&self, stack: &mut ImageStack) {
        match self {
            Op::StackOp(s) => s.apply(stack),
            Op::ImageOp(o) => o.apply(stack),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StackOp { Dup(usize), Drop, Over, Rot(usize) }

impl StackOp {
    fn parse(op: &str) -> Option<Self> {
        let split: Vec<&str> = op.split_whitespace().collect();
        match split[0] {
            "DUP" => {
                if split.len() == 1 {
                    Some(StackOp::Dup(1))
                } else {
                    Some(StackOp::Dup(split[1].parse().unwrap()))
                }
            },
            "DROP" => Some(StackOp::Drop),
            "OVER" => Some(StackOp::Over),
            "SWAP" => Some(StackOp::Rot(2)),
            "ROT" => {
                if split.len() == 1 {
                    Some(StackOp::Rot(3))
                } else {
                    Some(StackOp::Rot(split[1].parse().unwrap()))
                }
            },
            _ => None,
        }
    }

    fn apply(&self, stack: &mut ImageStack) {
        // TODO: inline stack function definitions here?
        match self {
            StackOp::Dup(n) => stack.dup(*n),
            StackOp::Drop => stack.drop(),
            StackOp::Over => stack.over(),
            StackOp::Rot(n) => stack.rot(*n),
        }
    }
}

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
}

impl ImageOp {
    pub fn apply(&self, stack: &mut ImageStack) {
        let result = match self {
            Self::Scale(s) => {
                let image = stack.pop();
                image.scale(*s)
            },
            Self::Gaussian(s) => {
                let image = stack.pop();
                image.gaussian(*s)
            },
            Self::Gray => {
                let image = stack.pop();
                image.grayscale()
            },
            Self::Rotate(t) => {
                let image = stack.pop();
                image.rotate(*t)
            },
            Self::Sobel => {
                let image = stack.pop();
                image.sobel()
            },
            Self::Carve(r) => {
                let image = stack.pop();
                image.carve(*r)
            },
            Self::AdaptiveThreshold(r) => {
                let image = stack.pop();
                image.adaptive_threshold(*r)
            },
            Self::OtsuThreshold => {
                let image = stack.pop();
                image.otsu_threshold()
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
            _ => None,
        }
    }
}

trait ImageExt {
    fn scale(&self, scale: f32) -> Self;
    fn gaussian(&self, sigma: f32) -> Self;
    fn rotate(&self, theta: f32) -> Self;
    fn sobel(&self) -> Self;
    fn carve(&self, ratio: f32) -> Self;
    fn adaptive_threshold(&self, block_radius: u32) -> Self;
    fn otsu_threshold(&self) -> Self;
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

impl ImageExt for DynamicImage {
    fn scale(&self, scale: f32) -> Self {
        let (w, h) = ((self.width() as f32 * scale) as u32, (self.height() as f32 * scale) as u32);
        self.resize(w, h, FilterType::Lanczos3)
    }

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


use image::{
    DynamicImage, DynamicImage::*, GenericImage, GenericImageView,
    Luma, LumaA, Rgb, Rgba, Bgr, Bgra, RgbaImage,
};
use imageproc::definitions::Clamp;
use crate::expr::Expr;
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
        let ch = &ch[..ch.len() - 1];
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
        "hcat" => if split.len() == 1 {
            Some(Box::new(Grid(2, 1)))
        } else {
            Some(Box::new(Grid(split[1].parse().unwrap(), 1)))
        },
        "vcat" => if split.len() == 1 {
            Some(Box::new(Grid(1, 2)))
        } else {
            Some(Box::new(Grid(1, split[1].parse().unwrap())))
        },
        "grid" => Some(Box::new(Grid(split[1].parse().unwrap(), split[2].parse().unwrap()))),
        "median" => Some(Box::new(Median(split[1].parse().unwrap(), split[2].parse().unwrap()))),
        "red" => Some(Box::new(Red)),
        "green" => Some(Box::new(Green)),
        "blue" => Some(Box::new(Blue)),
        "id" => Some(Box::new(Id)),
        "func" => Some(Box::new(parse_func(op))),
        "func2" => Some(Box::new(parse_func2(op))),
        "const" => Some(Box::new(parse_const(op))),
        "circle" => Some(Box::new(parse_circle(op))),
        _ => None,
    }
}

fn parse_circle(def: &str) -> Circle {
    // TODO: support drawing a centred circle without the caller having to specify the centre.
    // TODO: need a sensible standardised format for drawing functions.
    let def = &def[7..];
    let split: Vec<&str> = def.split_whitespace().collect();
    let fill = match split[0].to_lowercase().as_str() {
        "filled" => FillType::Filled,
        "hollow" => FillType::Hollow,
        _ => panic!("Invalid fill type"),
    };
    let center = (split[1].parse::<i32>().unwrap(), split[2].parse::<i32>().unwrap());
    let radius = split[3].parse::<i32>().unwrap();
    let vals: Vec<u8> = split[4..split.len()].iter().map(|v| v.trim()).map(|v| v.parse::<u8>().unwrap()).collect();
    let color = color_from_vals(&vals);
    Circle { fill, center, radius, color }
}

fn color_from_vals(vals: &[u8]) -> Color {
    match vals.len() {
        1 => Color::Luma(Luma([vals[0]])),
        2 => Color::LumaA(LumaA([vals[0], vals[1]])),
        3 => Color::Rgb(Rgb([vals[0], vals[1], vals[2]])),
        4 => Color::Rgba(Rgba([vals[0], vals[1], vals[2], vals[3]])),
        _ => panic!("Invalid color"),
    }
}

fn parse_const(def: &str) -> Const {
    let def = &def[6..];
    // Luma:
    //  const 100 200 (10)
    // LumaA:
    //  const 100 200 (10, 20)
    // Rgb:
    //  const 100 200 (50, 90, 100)
    // Rgba:
    //  const 100 200 (50, 90, 40, 20)
    // Bgr and Bgra not supported
    // TODO: parse operations like a sensible person would
    let split: Vec<&str> = def.split('(').collect();
    let dims: Vec<u32> = split[0].split_whitespace().map(|d| d.parse::<u32>().unwrap()).collect();
    let vals: Vec<u8> = split[1][..split[1].len() - 1].split(',').map(|v| v.trim()).map(|v| v.parse::<u8>().unwrap()).collect();
    let color = color_from_vals(&vals);

    Const {
        width: dims[0],
        height: dims[1],
        color
    }
}

fn parse_func(func: &str) -> Func {
    Func {
        text: func.into(),
        expr: crate::expr::parse_func(&func[5..]),
    }
}

fn parse_func2(func: &str) -> Func2 {
    Func2 {
        text: func.into(),
        expr: crate::expr::parse_func(&func[6..]),
    }
}

fn one_in_one_out<F>(stack: &mut ImageStack, f: F)
where
    F: FnOnce(&DynamicImage) -> DynamicImage
{
    let image = stack.pop();
    let result = f(&image);
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

/// Arrange images into a grid. First argument is the number of columns and second the number of rows.
#[derive(Debug)]
struct Grid(u32, u32);

impl ImageOp for Grid {
    fn apply(&self, stack: &mut ImageStack) {
        let images = stack.pop_n(self.0 as usize * self.1 as usize);
        let result = grid(&images, self.0, self.1);
        stack.push(result);
    }
}

fn grid(images: &[DynamicImage], cols: u32, rows: u32) -> DynamicImage {
    let (cols, rows) = (cols as usize, rows as usize);
    assert!(images.len() >= cols * rows);
    let images = &images[..cols * rows];
    // TODO: handle formats properly
    let images: Vec<_> = images.iter().map(|i| i.to_rgba()).collect();

    // Find the widest image in each column and the tallest image in each row
    let mut widths = Vec::with_capacity(cols);
    for c in 0..cols {
        let mut w = 0;
        for r in 0..rows {
            w = cmp::max(w, images[r * cols + c].width());
        }
        widths.push(w);
    }
    let mut heights = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut h = 0;
        for c in 0..cols {
            h = cmp::max(h, images[r * cols + c].height());
        }
        heights.push(h);
    }

    let lefts: Vec<_> = std::iter::once(0).chain(widths.iter().scan(0, |state, &x| { *state += x; Some(*state) })).collect();
    let tops: Vec<_> = std::iter::once(0).chain(heights.iter().scan(0, |state, &x| { *state += x; Some(*state) })).collect();

    let mut out = RgbaImage::new(widths.iter().sum(), heights.iter().sum());

    for r in 0..rows {
        for c in 0..cols {
            let image = &images[r * cols + c];
            out.copy_from(image, lefts[c], tops[r]);
        }
    }

    ImageRgba8(out)
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

/// The identity function. Sometimes helpful to make pipelines easier to write.
#[derive(Debug)]
struct Id;

impl ImageOp for Id {
    fn apply(&self, _: &mut ImageStack) {
        // Do nothing
    }
}

/// User defined per-subpixel function.
struct Func {
    /// The definition provided by the user, to use in logging.
    text: String,
    /// The expression to evalute per-subpixel.
    expr: Expr,
}

impl std::fmt::Debug for Func {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Func({})", self.text)
    }
}

impl ImageOp for Func {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| func(i, &self.expr));
    }
}

fn func(image: &DynamicImage, expr: &Expr) -> DynamicImage {
    let f = |p, x, y| {
        let r = expr.evaluate(x as f32, y as f32, p as f32, 0.0);
        <u8 as Clamp<f32>>::clamp(r)
    };
    match image {
        ImageLuma8(image) => ImageLuma8(map_subpixels_with_coords(image, f)),
        ImageLumaA8(image) => ImageLumaA8(map_subpixels_with_coords(image, f)),
        ImageRgb8(image) => ImageRgb8(map_subpixels_with_coords(image, f)),
        ImageRgba8(image) => ImageRgba8(map_subpixels_with_coords(image, f)),
        ImageBgr8(image) => ImageBgr8(map_subpixels_with_coords(image, f)),
        ImageBgra8(image) => ImageBgra8(map_subpixels_with_coords(image, f)),
    }
}


/// User defined per-subpixel function, taking two input images.
struct Func2 {
    /// The definition provided by the user, to use in logging.
    text: String,
    /// The expression to evalute per-subpixel.
    expr: Expr,
}

impl std::fmt::Debug for Func2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Func2({})", self.text)
    }
}

impl ImageOp for Func2 {
    fn apply(&self, stack: &mut ImageStack) {
        let image1 = stack.pop();
        let image2 = stack.pop();
        let result = func2(&image1, &image2, &self.expr);
        stack.push(result);
    }
}

fn func2(image1: &DynamicImage, image2: &DynamicImage, expr: &Expr) -> DynamicImage {
    let f = |p, q, x, y| {
        let r = expr.evaluate(x as f32, y as f32, p as f32, q as f32);
        <u8 as Clamp<f32>>::clamp(r)
    };
    // TODO: don't do unnecessary conversions everywhere. Rather than constantly converting
    // TODO: between formats or adding elaborate format checking, maybe we should just do all
    // TODO: calculations at RGBA.
    let image1 = image1.to_rgba();
    let image2 = image2.to_rgba();
    ImageRgba8(map_subpixels_with_coords2(&image1, &image2, f))
}

use imageproc::{definitions::Image, map::{ChannelMap, WithChannel}};
use image::{ImageBuffer, Pixel, Primitive};

// Could add this to imageproc, but either way we should extend the syntax for custom functions
// to support user-provided functions on entire pixels rather than just per channel.
#[allow(deprecated)]
fn map_subpixels_with_coords<I, P, F, S>(image: &I, f: F) -> Image<ChannelMap<P, S>>
where
    I: GenericImage<Pixel = P>,
    P: WithChannel<S> + 'static,
    S: Primitive + 'static,
    F: Fn(P::Subpixel, u32, u32) -> S,
{
    let (width, height) = image.dimensions();
    let mut out: ImageBuffer<ChannelMap<P, S>, Vec<S>> = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let out_channels = out.get_pixel_mut(x, y).channels_mut();
            for c in 0..P::channel_count() {
                out_channels[c as usize] = f(unsafe {
                    *image
                        .unsafe_get_pixel(x, y)
                        .channels()
                        .get_unchecked(c as usize)
                }, x, y);
            }
        }
    }

    out
}

#[allow(deprecated)]
fn map_subpixels_with_coords2<I, P, S, F>(image1: &I, image2: &I, f: F) -> Image<ChannelMap<P, S>>
where
    I: GenericImage<Pixel = P>,
    P: WithChannel<S> + 'static,
    S: Primitive + 'static,
    P::Subpixel: std::fmt::Debug,
    F: Fn(P::Subpixel, P::Subpixel, u32, u32) -> S,
{
    // TODO: allow differently sized images, do sensible format conversions when the two images have different formats
    assert_eq!(image1.dimensions(), image2.dimensions());

    let (width, height) = image1.dimensions();
    let mut out: ImageBuffer<ChannelMap<P, S>, Vec<S>> = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let out_channels = out.get_pixel_mut(x, y).channels_mut();
            for c in 0..P::channel_count() {
                let p = unsafe {
                    *image1
                        .unsafe_get_pixel(x, y)
                        .channels()
                        .get_unchecked(c as usize)
                };
                let q = unsafe {
                    *image2
                        .unsafe_get_pixel(x, y)
                        .channels()
                        .get_unchecked(c as usize)
                };
                out_channels[c as usize] = f(p, q, x, y);
            }
        }
    }

    out
}

/// Create an image with a single constant value.
#[derive(Debug)]
struct Const {
    width: u32,
    height: u32,
    color: Color,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ColorSpace {
    Luma, LumaA, Rgb, Rgba, Bgr, Bgra
}

#[derive(Debug, Clone)]
enum Color {
    Luma(Luma<u8>),
    LumaA(LumaA<u8>),
    Rgb(Rgb<u8>),
    Rgba(Rgba<u8>),
    Bgr(Bgr<u8>),
    Bgra(Bgra<u8>),
}

impl Color {
    fn color_space(&self) -> ColorSpace {
        match self {
            Color::Luma(_) => ColorSpace::Luma,
            Color::LumaA(_) => ColorSpace::LumaA,
            Color::Rgb(_) => ColorSpace::Rgb,
            Color::Rgba(_) => ColorSpace::Rgba,
            Color::Bgr(_) => ColorSpace::Bgr,
            Color::Bgra(_) => ColorSpace::Bgra,
        }
    }
}

fn color_space(image: &DynamicImage) -> ColorSpace {
    match image {
        ImageLuma8(_) => ColorSpace::Luma,
        ImageLumaA8(_) => ColorSpace::LumaA,
        ImageRgb8(_) => ColorSpace::Rgb,
        ImageRgba8(_) => ColorSpace::Rgba,
        ImageBgr8(_) => ColorSpace::Bgr,
        ImageBgra8(_) => ColorSpace::Bgra,
    }
}

impl ImageOp for Const {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |_| constant(self));
    }
}

fn constant(c: &Const) -> DynamicImage {
    match c.color {
        Color::Luma(l) => ImageLuma8(ImageBuffer::from_pixel(c.width, c.height, l)),
        Color::LumaA(l) => ImageLumaA8(ImageBuffer::from_pixel(c.width, c.height, l)),
        Color::Rgb(l) => ImageRgb8(ImageBuffer::from_pixel(c.width, c.height, l)),
        Color::Rgba(l) => ImageRgba8(ImageBuffer::from_pixel(c.width, c.height, l)),
        Color::Bgr(l) => ImageBgr8(ImageBuffer::from_pixel(c.width, c.height, l)),
        Color::Bgra(l) => ImageBgra8(ImageBuffer::from_pixel(c.width, c.height, l)),
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FillType { Filled, Hollow }

/// Draw a circle with the given center, radius, color, and fill type.
#[derive(Debug)]
struct Circle {
    fill: FillType,
    center: (i32, i32),
    radius: i32,
    color: Color,
}

impl ImageOp for Circle {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| draw_circle(i, self));
    }
}

// TODO: if we always pass the op type itself then we can standardise the calls to one_in_one_out across all ops.
fn draw_circle(image: &DynamicImage, circle: &Circle) -> DynamicImage {
    use imageproc::drawing::{draw_hollow_circle, draw_filled_circle};
    // TODO: Handle formats properly - choose the "most general" color space.
    let image = image.to_rgba();
    let color = match circle.color {
        Color::Luma(c) => c.to_rgba(),
        Color::LumaA(c) => c.to_rgba(),
        Color::Rgb(c) => c.to_rgba(),
        Color::Rgba(c) => c.to_rgba(),
        Color::Bgr(c) => c.to_rgba(),
        Color::Bgra(c) => c.to_rgba(),
    };

    // TODO: We always consume entries from the stack, so we using mutating functions where
    // TODO: possible, rather than just constantly allocating new images.
    match circle.fill {
        FillType::Filled => ImageRgba8(draw_filled_circle(&image, circle.center, circle.radius, color)),
        FillType::Hollow => ImageRgba8(draw_hollow_circle(&image, circle.center, circle.radius, color)),
    }
}
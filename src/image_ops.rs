
use image::{
    DynamicImage, DynamicImage::*, GenericImage, GenericImageView,
    Luma, LumaA, Rgb, Rgba, Bgr, Bgra, RgbaImage,
};
use imageproc::definitions::Clamp;
use crate::expr::Expr;
use std::cmp;
use crate::ImageStack;
use crate::parse_utils::{op_zero, op_one, op_one_opt, op_two, int};
use nom::{
    IResult,
    branch::alt,
    bytes::complete::tag,
    combinator::{all_consuming, map},
    character::complete::{space0, space1},
    multi::separated_nonempty_list,
    number::complete::float,
    sequence::{delimited, pair, preceded, tuple}
};

pub fn run_pipeline(pipeline: &str, inputs: Vec<DynamicImage>, verbose: bool) -> Vec<DynamicImage> {
    let mut stack = ImageStack::new(inputs);
    let ops = parse(pipeline);

    for op in ops {
        if verbose {
            println!("Applying {:?}", op);
        }
        op.apply(&mut stack);
    }

    stack.contents()
}

/// An image processing operation that operates on a stack of images.
pub trait ImageOp : std::fmt::Debug {
    fn apply(&self, stack: &mut ImageStack);
}

pub fn parse(pipeline: &str) -> Vec<Box<dyn ImageOp>> {
    let parsed = parse_pipeline(pipeline);
    match parsed {
        Ok(p) => p.1,
        Err(e) => {
            let remaining = match e {
                nom::Err::Error(e) => e.0,
                _ => unreachable!(),
            };
            let consumed = &pipeline[0..pipeline.len() - remaining.len()];
            panic!(
                "
Unable to parse pipeline.

Consumed: '{}'
Remaining: '{}'

The error is likely near the start of the remaining (unparsed) text.
",
                consumed,
                remaining
            )
        }
    }
}

fn parse_pipeline(input: &str) -> IResult<&str, Vec<Box<dyn ImageOp>>> {
    all_consuming(
        separated_nonempty_list(parse_connector, parse_image_op)
    )(input)
}

fn parse_connector(input: &str) -> IResult<&str, ()> {
    delimited(space0, map(tag(">"), |_| ()), space0)(input)
}

/// I wanted to write this as a function, but got confused by the resulting compiler errors.
macro_rules! map_box {
    ($parser:expr) => {
        map($parser, |o| {
            let boxed: Box<dyn ImageOp> = Box::new(o);
            boxed
        })
    }
}

fn parse_image_op(input: &str) -> IResult<&str, Box<dyn ImageOp>> {
    // Nested alts are required because alt only supports tuples with up 21 elements.
    alt((
        alt((
            map_box!(parse_array),
            map_box!(op_one("athresh", int::<u32>, |x| AdaptiveThreshold(x))),
            map_box!(op_zero("blue", Blue)),
            map_box!(op_one("carve", float, |x| Carve(x))),
            map_box!(parse_circle),
            map_box!(parse_const),
            map_box!(op_zero("DROP", Drop)),
            map_box!(op_one_opt("DUP", int::<usize>, |x| Dup(x.unwrap_or(1)))),
            map_box!(parse_func),
            map_box!(parse_func2),
            map_box!(op_one("gaussian", float, |x| Gaussian(x))),
            map_box!(op_zero("gray", Gray)),
            map_box!(op_zero("green", Green)),
            map_box!(op_two("grid", int::<u32>, int::<u32>, |w, h| Grid(w, h))),
            map_box!(op_one_opt("hcat", int::<u32>, |x| Grid(x.unwrap_or(2), 1))),
            map_box!(op_zero("id", Id)),
            map_box!(op_two("median", int::<u32>, int::<u32>, |rx, ry| Median(rx, ry))),
            map_box!(op_zero("othresh", OtsuThreshold)),
            map_box!(op_zero("red", Red)),
            map_box!(op_one_opt("ROT", int::<usize>, |x| Rot(x.unwrap_or(3)))),
            map_box!(op_one("rotate", float, |x| Rotate(x))),
        )),
        alt((
            map_box!(op_one("scale", float, |x| Scale(x))),
            map_box!(op_zero("sobel", Sobel)),
            map_box!(op_zero("SWAP", Rot(2))),
            map_box!(op_two("translate", int::<i32>, int::<i32>, |tx, ty| Translate(tx, ty))),
            map_box!(op_one_opt("vcat", int::<u32>, |x| Grid(1, x.unwrap_or(2)))),
        )),
    ))(input)
}

fn parse_array(input: &str) -> IResult<&str, Array> {
    map(
        delimited(
            tag("["),
            separated_nonempty_list(
                delimited(space0, tag(","), space0),
                parse_image_op
            ),
            tag("]")
        ),
        |v| Array(v)
    )(input)
}

// TODO: Remove duplication between this and parse_const
// circle filltype cx cy radius (color)
// Uses the same colour format as const
fn parse_circle(input: &str) -> IResult<&str, Circle> {
    map(
        preceded(
            tag("circle"),
            tuple((
                preceded(space1, alt((tag("filled"), tag("hollow")))),
                pair(preceded(space1, int::<i32>), preceded(space1, int::<i32>)),
                preceded(space1, int::<i32>),
                preceded(space1, parse_color),
            ))
        ),
        |(fill, center, radius, color)| Circle { fill: fill.into(), center, radius, color }
    )(input)
}

fn parse_color(input: &str) -> IResult<&str, Color> {
    map(
        delimited(
            tag("("),
            separated_nonempty_list(
                delimited(space0, tag(","), space0),
                int::<u8>
            ),
            tag(")")
        ),
        |vs| color_from_vals(&vs)
    )(input)
}

// Luma:
//  const 100 200 (10)
// LumaA:
//  const 100 200 (10, 20)
// Rgb:
//  const 100 200 (50, 90, 100)
// Rgba:
//  const 100 200 (50, 90, 40, 20)
// Bgr and Bgra not supported
fn parse_const(input: &str) -> IResult<&str, Const> {
    map(
        preceded(
            tag("const"),
            tuple((
                preceded(space1, int::<u32>),
                preceded(space1, int::<u32>),
                preceded(space1, parse_color)
            ))
        ),
        |(width, height, color)| Const { width, height, color }
    )(input)
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

fn parse_func(input: &str) -> IResult<&str, Func> {
    let (i, (text, expr)) = crate::expr::parse_func(input, "func")?;
    Ok((i, Func { text, expr }))
}

fn parse_func2(input: &str) -> IResult<&str, Func2> {
    let (i, (text, expr)) = crate::expr::parse_func(input, "func2")?;
    Ok((i, Func2 { text, expr }))
}

fn one_in_one_out<F>(stack: &mut ImageStack, f: F)
where
    F: FnOnce(&DynamicImage) -> DynamicImage
{
    let image = stack.pop();
    let result = f(&image);
    stack.push(result);
}

/// Duplicates the top element of the stack n times.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Dup(usize);

impl ImageOp for Dup {
    fn apply(&self, stack: &mut ImageStack) {
        stack.dup(self.0);
    }
}

/// Discards the top element of the stack.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Drop;

impl ImageOp for Drop {
    fn apply(&self, stack: &mut ImageStack) {
        stack.drop();
    }
}

/// Rotates the top n elements of the stack by 1.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Rot(usize);

impl ImageOp for Rot {
    fn apply(&self, stack: &mut ImageStack) {
        stack.rot(self.0);
    }
}

/// Scale both width and height by given multiplier.
#[derive(Debug, Copy, Clone, PartialEq)]
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
#[derive(Debug, Copy, Clone, PartialEq)]
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Gray;

impl ImageOp for Gray {
    fn apply(&self, stack: &mut ImageStack) {
        let image = stack.pop();
        let result = gray(image);
        stack.push(result);
    }
}

fn gray(image: DynamicImage) -> DynamicImage {
    match image {
        ImageLuma8(i) => ImageLuma8(i),
        _ => image.grayscale(),
    }
}

 /// Rotate clockwise about the image's center by the given angle in degrees.
#[derive(Debug, Copy, Clone, PartialEq)]
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
#[derive(Debug, Copy, Clone, PartialEq)]
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct OtsuThreshold;

impl ImageOp for OtsuThreshold {
    fn apply(&self, stack: &mut ImageStack) {
        let image = stack.pop();
        let result = otsu_threshold(image);
        stack.push(result);
    }
}

fn otsu_threshold(image: DynamicImage) -> DynamicImage {
    use imageproc::contrast::{otsu_level, threshold_mut};
    let mut image = match gray(image) {
        ImageLuma8(i) => i,
        _ => unreachable!(),
    };
    let level = otsu_level(&image);
    threshold_mut(&mut image, level);
    ImageLuma8(image)
}

/// Arrange images into a grid. First argument is the number of columns and second the number of rows.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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

// TODO: Over-writing with const means you have to do some awkward stack manipulation
// TODO: if you want to combine the constant image with your inputs.
// TODO: Maybe it would be better to allow constant images as 'pseudo-inputs', which get
// TODO: injected at the start of the pipeline, rather than manipulating the existing stack.

/// Create an image with a single constant value.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Const {
    width: u32,
    height: u32,
    color: Color,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Color {
    Luma(Luma<u8>),
    LumaA(LumaA<u8>),
    Rgb(Rgb<u8>),
    Rgba(Rgba<u8>),
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
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FillType { Filled, Hollow }

impl From<&str> for FillType {
    fn from(fill: &str) -> Self {
        match fill {
            "filled" => FillType::Filled,
            "hollow" => FillType::Hollow,
            _ => panic!("Invalid FillType"),
        }
    }
}

/// Draw a circle with the given center, radius, color, and fill type.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Circle {
    fill: FillType,
    center: (i32, i32),
    radius: i32,
    color: Color,
}

impl ImageOp for Circle {
    fn apply(&self, stack: &mut ImageStack) {
        let image = stack.pop();
        let result = draw_circle(image, self);
        stack.push(result);
    }
}

fn draw_circle(image: DynamicImage, circle: &Circle) -> DynamicImage {
    use imageproc::drawing::{draw_hollow_circle_mut, draw_filled_circle_mut};
    // TODO: Handle formats properly - choose the "most general" color space.
    let mut image = image.to_rgba();
    let color = match circle.color {
        Color::Luma(c) => c.to_rgba(),
        Color::LumaA(c) => c.to_rgba(),
        Color::Rgb(c) => c.to_rgba(),
        Color::Rgba(c) => c.to_rgba(),
    };
    match circle.fill {
        FillType::Filled => draw_filled_circle_mut(&mut image, circle.center, circle.radius, color),
        FillType::Hollow => draw_hollow_circle_mut(&mut image, circle.center, circle.radius, color),
    };
    ImageRgba8(image)
}

/// Translate the image.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Translate(i32, i32);

impl ImageOp for Translate {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| translate(i, self.0, self.1));
    }
}

fn translate(image: &DynamicImage, tx: i32, ty: i32) -> DynamicImage {
    use imageproc::geometric_transformations::translate;
    let t = (tx, ty);
    match image {
        ImageLuma8(image) => ImageLuma8(translate(image, t)),
        ImageLumaA8(image) => ImageLumaA8(translate(image, t)),
        ImageRgb8(image) => ImageRgb8(translate(image, t)),
        ImageRgba8(image) => ImageRgba8(translate(image, t)),
        ImageBgr8(image) => ImageBgr8(translate(image, t)),
        ImageBgra8(image) => ImageBgra8(translate(image, t)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_pipeline_parse(pipeline: &str, expected: &[&str]) {
        let parsed = parse_pipeline(pipeline);
        match parsed {
            Err(_) => panic!("Parse should succeed"),
            Ok((i, v)) => {
                assert_eq!(i, "");
                let descriptions: Vec<String> = v.iter().map(|o| format!("{:?}", o)).collect();
                assert_eq!(descriptions, expected);
            }
        }
    }

    fn assert_pipeline_parse_failure(pipeline: &str) {
        let parsed = parse_pipeline(pipeline);
        match parsed {
            Ok((i, v)) => panic!("Parse succeeded, but should have failed. i: {:?}, v: {:?}", i, v),
            Err(_) => (), // TODO: add error reporting, with tests
        }
    }

    #[test]
    fn test_parse_valid_single_stage_pipeline() {
        assert_pipeline_parse(
            "circle filled 231 337 100 (255, 255, 0)",
            &vec!["Circle { fill: Filled, center: (231, 337), radius: 100, color: Rgb(Rgb([255, 255, 0])) }"]
        );
    }

    #[test]
    fn test_parse_invalid_single_stage_pipeline() {
        // Not valid, as grid requires two numerical inputs)
        assert_pipeline_parse_failure("grid 1");
    }

    #[test]
    fn test_parse_valid_multi_stage_pipeline() {
        assert_pipeline_parse(
            "grid 1 2 > gray > scale 3 > id",
            &vec!["Grid(1, 2)", "Gray", "Scale(3.0)", "Id"]
        );
    }

    #[test]
    fn test_parse_invalid_multi_stage_pipeline() {
        // The grid stage is invalid
        assert_pipeline_parse_failure("gray > grid 1 > scale 3 > id");
    }

    #[test]
    fn test_parse_circle() {
        assert_pipeline_parse(
            "circle filled 231 337 100 (255, 255, 0)",
            &vec!["Circle { fill: Filled, center: (231, 337), radius: 100, color: Rgb(Rgb([255, 255, 0])) }"]
        );
    }

    #[test]
    fn test_parse_hcat() {
        // No width provided - default to 2
        assert_pipeline_parse(
            "hcat",
            &vec!["Grid(2, 1)"]
        );
        // Width provided
        assert_pipeline_parse(
            "hcat 3",
            &vec!["Grid(3, 1)"]
        );
    }

    #[test]
    fn test_parse_scale() {
        // No number provided
        assert_pipeline_parse_failure("scale");
        // Correct incantation, integer literal
        assert_pipeline_parse(
            "scale 3",
            &vec!["Scale(3.0)"]
        );
        // Correct incantation, floating point literal
        assert_pipeline_parse(
            "scale 3.0",
            &vec!["Scale(3.0)"]
        );
    }

    #[test]
    fn test_parse_gray() {
        assert_pipeline_parse(
            "gray",
            &vec!["Gray"]
        );
    }

    #[test]
    fn test_parse_grid() {
        // Only one number provided
        assert_pipeline_parse_failure("grid 12");
        // Correct incantation
        assert_pipeline_parse(
            "grid 12 34",
            &vec!["Grid(12, 34)"]
        );
    }

    use test::{Bencher, black_box};

    #[bench]
    fn bench_pipeline_parsing(b: &mut Bencher) {
        let pipeline = black_box(
            "gray > func { 255 * (p > 100) } > rotate 45 > othresh > scale 2"
        );
        b.iter(|| {
            let pipeline = parse_pipeline(pipeline).unwrap();
            black_box(pipeline);
        });
    }

    #[bench]
    fn bench_run_pipeline_with_user_defined_func(b: &mut Bencher) {
        let pipeline = "func { 255 * (p > 100) }";
        let image = DynamicImage::ImageLuma8(
            ImageBuffer::from_fn(100, 100, |x, y| Luma([(x + y) as u8]))
        );
        b.iter(|| {
            let inputs = black_box(vec![image.clone()]);
            let _ = black_box(run_pipeline(pipeline, inputs, false));
        });
    }

    #[bench]
    fn bench_run_pipeline(b: &mut Bencher) {
        let pipeline = "gray > DUP > rotate 45 > ROT 2 > othresh > hcat";
        let image = DynamicImage::ImageLuma8(
            ImageBuffer::from_fn(100, 100, |x, y| Luma([(x + y) as u8]))
        );
        b.iter(|| {
            let inputs = black_box(vec![image.clone()]);
            let _ = black_box(run_pipeline(pipeline, inputs, false));
        });
    }
}

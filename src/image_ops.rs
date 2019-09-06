
use image::{
    DynamicImage, DynamicImage::*, GenericImage, GenericImageView,
    Luma, LumaA, Rgb, Rgba, Bgr, Bgra, RgbaImage,
};
use imageproc::definitions::Clamp;
use crate::expr::Expr;
use std::cmp;
use crate::ImageStack;
use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag, tag_no_case},
    combinator::{all_consuming, map, map_res, opt},
    character::complete::{digit1, space0, space1},
    multi::separated_nonempty_list,
    number::complete::float,
    sequence::{delimited, pair, preceded, tuple}
};

/// An image processing operation that operates on a stack of images.
pub trait ImageOp : std::fmt::Debug {
    fn apply(&self, stack: &mut ImageStack);
}

pub fn parse(pipeline: &str) -> Vec<Box<dyn ImageOp>> {
    let pipeline = parse_pipeline(pipeline);
    match pipeline {
        Ok(p) => p.1,
        Err(e) => {
            panic!("Unable to parse pipeline: {:?}", e)
        }
    }
}

fn parse_pipeline(input: &str) -> IResult<&str, Vec<Box<dyn ImageOp>>> {
    all_consuming(
        separated_nonempty_list(parse_connector, parse_image_op)
    )(input)
}

fn parse_connector(input: &str) -> IResult<&str, ()> {
    delimited(space0, map(tag(">"), |s: &str| ()), space0)(input)
}

/// I wanted to write this as a function, but got confused by the resulting compiler errors.
macro_rules! box_output {
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
            box_output!(parse_adaptive_threshold),
            box_output!(parse_array),
            box_output!(parse_blue),
            box_output!(parse_carve),
            box_output!(parse_circle),
            box_output!(parse_const),
            box_output!(parse_drop),
            box_output!(parse_dup),
            box_output!(parse_func),
            box_output!(parse_func2),
            box_output!(parse_gaussian),
            box_output!(parse_gray),
            box_output!(parse_green),
            box_output!(parse_grid),
            box_output!(parse_hcat),
            box_output!(parse_id),
            box_output!(parse_median),
            box_output!(parse_othresh),
            box_output!(parse_red),
            box_output!(parse_rot),
            box_output!(parse_rotate),
        )),
        alt((
            box_output!(parse_scale),
            box_output!(parse_sobel),
            box_output!(parse_swap),
            box_output!(parse_translate),
            box_output!(parse_vcat),
        )),
    ))(input)
}

// TODO: remove duplication between this and parse_pipeline
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

fn int<T: std::str::FromStr>(input: &str) -> IResult<&str, T> {
    map_res(digit1, |s: &str| s.parse::<T>())(input)
}

/// Generates parsers which match a literal and return a struct of the specific field-less type.
macro_rules! named_empty_struct_parsers {
    ($( $parser_name:ident, $name:expr, $type:ident );* $(;)? )  => {
        $(
            fn $parser_name(input: &str) -> IResult<&str, $type> {
                map(tag_no_case($name), |_| $type)(input)
            }
        )*
    }
}

named_empty_struct_parsers!(
    parse_gray, "gray", Gray;
    parse_id, "id", Id;
    parse_red, "red", Red;
    parse_green, "green", Green;
    parse_blue, "blue", Blue;
    parse_sobel, "sobel", Sobel;
    parse_othresh, "othresh", OtsuThreshold;
    parse_drop, "DROP", Drop;
);

// Operator which takes no args
fn op_zero<'a, 'b, T: Clone>(input: &'a str, name: &'b str, t: T) -> IResult<&'a str, T> {
    map(tag_no_case(name), |_| t.clone())(input)
}

// Operator which takes a single arg
fn op_one<'a, 'b, T, F1, A1, G>(
    input: &'a str,
    name: &'b str,
    arg1: F1,
    build: G
) -> IResult<&'a str, T>
where
    F1: Fn(&'a str) -> IResult<&'a str, A1>,
    G: Fn(A1) -> T
{
    map(
        preceded(
            tag_no_case(name),
            preceded(space1, arg1)
        ),
        build
    )(input)
}

// Operator which takes a single optional arg
// TODO: unify with op_one
fn op_one_opt<'a, 'b, T, F1, A1, G>(
    input: &'a str,
    name: &'b str,
    arg1: F1,
    build: G
) -> IResult<&'a str, T>
where
    F1: Fn(&'a str) -> IResult<&'a str, A1>,
    G: Fn(Option<A1>) -> T
{
    map(
        preceded(
            tag_no_case(name),
            opt(preceded(space1, arg1))
        ),
        build
    )(input)
}

// Operator which takes twp args
fn op_two<'a, 'b, T, F1, F2, A1, A2, G>(
    input: &'a str,
    name: &'b str,
    arg1: F1,
    arg2: F2,
    build: G
) -> IResult<&'a str, T>
where
    F1: Fn(&'a str) -> IResult<&'a str, A1>,
    F2: Fn(&'a str) -> IResult<&'a str, A2>,
    G: Fn(A1, A2) -> T
{
    map(
        preceded(
            tag_no_case(name),
            tuple((
                preceded(space1, arg1),
                preceded(space1, arg2),
            ))
        ),
        |(val1, val2)| build(val1, val2)
    )(input)
}

// 'grid 12 34' -> Grid(12, 34)
fn parse_grid(input: &str) -> IResult<&str, Grid> {
    op_two(input, "grid", int::<u32>, int::<u32>, |w, h| Grid(w, h))
}

// 'median 12 34' -> Median(12, 34)
fn parse_median(input: &str) -> IResult<&str, Median> {
    op_two(input, "median", int::<u32>, int::<u32>, |rx, ry| Median(rx, ry))
}

// 'translate 10 20' -> Translate(10, 20)
fn parse_translate(input: &str) -> IResult<&str, Translate> {
    op_two(input, "translate", int::<i32>, int::<i32>, |tx, ty| Translate(tx, ty))
}

// 'scale 12' -> Scale(12.0)
fn parse_scale(input: &str) -> IResult<&str, Scale> {
    op_one(input, "scale", float, |x| Scale(x))
}

// 'gaussian 12' -> Gaussian(12.0)
fn parse_gaussian(input: &str) -> IResult<&str, Gaussian> {
    op_one(input, "gaussian", float, |x| Gaussian(x))
}

// 'rotate 45' -> Rotate(45.0)
fn parse_rotate(input: &str) -> IResult<&str, Rotate> {
    op_one(input, "rotate", float, |x| Rotate(x))
}

// 'carve 0.85' -> Carve(0.85)
fn parse_carve(input: &str) -> IResult<&str, Carve> {
    op_one(input, "carve", float, |x| Carve(x))
}

// 'athresh 12' -> AdaptiveThreshold(12)
fn parse_adaptive_threshold(input: &str) -> IResult<&str, AdaptiveThreshold> {
    op_one(input, "athresh", int::<u32>, |x| AdaptiveThreshold(x))
}

// 'DUP' -> Dup(1)
// 'DUP 3' -> Dup(3)
fn parse_dup(input: &str) -> IResult<&str, Dup> {
    op_one_opt(input, "dup", int::<usize>, |x| Dup(x.unwrap_or(1)))
}

// 'ROT' -> Rot(3)
// 'ROT 4' -> Rot(4)
fn parse_rot(input: &str) -> IResult<&str, Rot> {
    op_one_opt(input, "rot", int::<usize>, |x| Rot(x.unwrap_or(3)))
}

// 'SWAP' -> Rot(2)
fn parse_swap(input: &str) -> IResult<&str, Rot> {
    op_zero(input, "swap", Rot(2))
}

// 'hcat' -> Grid 2 1
// 'hcat 3' -> Grid 3 1
fn parse_hcat(input: &str) -> IResult<&str, Grid> {
    op_one_opt(input, "hcat", int::<u32>, |x| Grid(x.unwrap_or(2), 1))
}

// 'vcat' -> Grid 1 2
// 'vcat 3' -> Grid 1 3
fn parse_vcat(input: &str) -> IResult<&str, Grid> {
    op_one_opt(input, "vcat", int::<u32>, |x| Grid(1, x.unwrap_or(2)))
}

// TODO: Remove duplication between this and parse_const
// circle filltype cx cy radius (color)
// Uses the same colour format as const
fn parse_circle(input: &str) -> IResult<&str, Circle> {
    map(
        preceded(
            tag_no_case("circle"),
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
            tag_no_case("const"),
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
        one_in_one_out(stack, |i| i.grayscale());
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
        one_in_one_out(stack, otsu_threshold);
    }
}

fn otsu_threshold(image: &DynamicImage) -> DynamicImage {
    use imageproc::contrast::{otsu_level, threshold};
    let gray = image.to_luma();
    ImageLuma8(threshold(&gray, otsu_level(&gray)))
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ColorSpace {
    Luma, LumaA, Rgb, Rgba, Bgr, Bgra
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    use nom::error::ErrorKind;

    #[test]
    fn test_parse_valid_single_stage_pipeline() {
        let pipeline = "circle filled 231 337 100 (255, 255, 0)";
        let parsed = parse_pipeline(pipeline);
        match parsed {
            Err(_) => panic!("Parse should succeed"),
            Ok((i, v)) => {
                assert_eq!(i, "");
                let descriptions: Vec<String> = v.iter().map(|o| format!("{:?}", o)).collect();
                let expected = vec![
                    String::from(
                        "Circle { fill: Filled, center: (231, 337), radius: 100, color: Rgb(Rgb([255, 255, 0])) }"
                    )
                ];
                assert_eq!(descriptions, expected);
            }
        }
    }

    #[test]
    fn test_parse_invalid_single_stage_pipeline() {
        let pipeline = "grid 1"; // Not valid, as grid requires two numerical inputs
        let parsed = parse_pipeline(pipeline);
        match parsed {
            Ok((i, v)) => panic!("Parse succeeded, but should have failed. i: {:?}, v: {:?}", i, v),
            Err(_) => (), // TODO: add error reporting, with tests
        }
    }

    #[test]
    fn test_parse_valid_multi_stage_pipeline() {
        let pipeline = "grid 1 2 > gray > scale 3 > id";
        let parsed = parse_pipeline(pipeline);
        match parsed {
            Err(_) => panic!("Parse should succeed"),
            Ok((i, v)) => {
                assert_eq!(i, "");
                let descriptions: Vec<String> = v.iter().map(|o| format!("{:?}", o)).collect();
                let expected: Vec<String> = vec!["Grid(1, 2)", "Gray", "Scale(3.0)", "Id"]
                    .into_iter()
                    .map(|s| s.into())
                    .collect();
                assert_eq!(descriptions, expected);
            }
        }
    }

    #[test]
    fn test_parse_invalid_multi_stage_pipeline() {
        let pipeline = "gray > grid 1 > scale 3 > id"; // The grid stage is invalid
        let parsed = parse_pipeline(pipeline);
        match parsed {
            Ok((i, v)) => panic!("Parse succeeded, but should have failed. i: {:?}, v: {:?}", i, v),
            Err(_) => (), // TODO: add error reporting, with tests
        }
    }

    #[test]
    fn test_parse_circle() {
        // RGB
        assert_eq!(
            parse_circle("circle filled 231 337 100 (255, 255, 0)"),
            Ok((
                "",
                Circle {
                    fill: FillType::Filled,
                    center: (231, 337),
                    radius: 100,
                    color: Color::Rgb(Rgb([255, 255, 0]))
                }
            ))
        );
    }

    #[test]
    fn test_parse_hcat() {
        // No width provided - default to 2
        assert_eq!(
            parse_hcat("hcat"),
            Ok(("", Grid(2, 1)))
        );
        // Width provided
        assert_eq!(
            parse_hcat("hcat 3"),
            Ok(("", Grid(3, 1)))
        );
    }

    #[test]
    fn test_parse_scale() {
        // Literal doesn't match
        assert_eq!(
            parse_scale("circle 1 2 3"),
            Err(nom::Err::Error(("circle 1 2 3", ErrorKind::Tag)))
        );
        // No number provided
        assert_eq!(
            parse_scale("scale"),
            Err(nom::Err::Error(("", ErrorKind::Space)))
        );
        // Correct incantation, integer literal
        assert_eq!(
            parse_scale("scale 3"),
            Ok(("", Scale(3.0)))
        );
        // Correct incantation, floating point literal
        assert_eq!(
            parse_scale("scale 3.0"),
            Ok(("", Scale(3.0)))
        );
    }

    #[test]
    fn test_parse_gray() {
        // Literal doesn't match
        assert_eq!(
            parse_gray("blue"),
            Err(nom::Err::Error(("blue", ErrorKind::Tag)))
        );
        // Literal matches
        assert_eq!(
            parse_gray("gray"),
            Ok(("", Gray))
        );
    }

    #[test]
    fn test_parse_grid() {
        // Literal doesn't match
        assert_eq!(
            parse_grid("circle 1 2 3"),
            Err(nom::Err::Error(("circle 1 2 3", ErrorKind::Tag)))
        );
        // Only one number provided
        assert_eq!(
            parse_grid("grid 12"),
            Err(nom::Err::Error(("", ErrorKind::Space)))
        );
        // Correct incantation
        assert_eq!(
            parse_grid("grid 12 34"),
            Ok(("", Grid(12, 34)))
        );
    }
}

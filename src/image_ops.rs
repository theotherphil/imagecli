
use image::{
    DynamicImage, DynamicImage::*, GenericImage, GenericImageView,
    Luma, LumaA, Rgb, Rgba, Bgr, Bgra, RgbaImage,
};
use imageproc::definitions::Clamp;
use crate::expr::Expr;
use std::cmp;
use crate::ImageStack;
use crate::parse_utils::{
    op_zero, op_one, op_one_opt, op_two, int, nonempty_sequence, named_arg
};
use nom::{
    IResult,
    branch::{alt, permutation},
    bytes::complete::tag,
    combinator::{all_consuming, map},
    character::complete::{space1},
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
    all_consuming(nonempty_sequence(">", parse_image_op))(input)
}

macro_rules! map_box { ($parser:expr) => { $crate::map_to_boxed_trait!($parser, ImageOp) } }

fn parse_image_op(input: &str) -> IResult<&str, Box<dyn ImageOp>> {
    // Nested alts are required because alt only supports tuples with up 21 elements.
    alt((
        alt((
            Array::parse,
            AdaptiveThreshold::parse,
            Blue::parse,
            Carve::parse,
            Circle::parse,
            Const::parse,
            Drop::parse,
            Dup::parse,
            Func::parse,
            Func2::parse,
            Gaussian::parse,
            Gray::parse,
            Green::parse,
            Grid::parse,
            map_box!(op_one_opt("hcat", int::<u32>, |x| Grid(x.unwrap_or(2), 1))),
            HFlip::parse,
            Id::parse,
            Median::parse,
            OtsuThreshold::parse,
            Red::parse,
            Resize::parse,
        )),
        alt((
            Rot::parse,
            Rotate::parse,
            Scale::parse,
            Sobel::parse,
            map_box!(op_zero("SWAP", Rot(2))),
            Translate::parse,
            map_box!(op_one_opt("vcat", int::<u32>, |x| Grid(1, x.unwrap_or(2)))),
            VFlip::parse,
        )),
    ))(input)
}

macro_rules! dynamic_map {
    ($dynimage:expr, $func:expr) => {
        match $dynimage {
            DynamicImage::ImageLuma8(image) => DynamicImage::ImageLuma8($func(image)),
            DynamicImage::ImageLumaA8(image) => DynamicImage::ImageLumaA8($func(image)),
            DynamicImage::ImageRgb8(image) => DynamicImage::ImageRgb8($func(image)),
            DynamicImage::ImageRgba8(image) => DynamicImage::ImageRgba8($func(image)),
            DynamicImage::ImageBgr8(image) => DynamicImage::ImageBgr8($func(image)),
            DynamicImage::ImageBgra8(image) => DynamicImage::ImageBgra8($func(image)),
        }
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

// TODO: this isn't object-safe, so a trait-based approach won't work.
// TODO: Maybe I just want to add a parse method to the type, and generate a function that returns
// TODO: a Description { type_name, usage, explanation } for this type.
pub trait ImageOpParse : ImageOp {
    fn usage() -> &'static str;
    fn explanation() -> &'static str;
    fn parse<'a>(input: &'a str) -> IResult<&'a str, Box<dyn ImageOp>>;
}

macro_rules! impl_image_op_parse {
    ($name:ident, $usage:expr, $explanation:expr, $parse:expr) => {
        impl ImageOpParse for $name {
            fn usage() -> &'static str { $usage }
            fn explanation() -> &'static str { $explanation }
            fn parse<'a>(input: &'a str) -> IResult<&'a str, Box<dyn ImageOp>> {
                map_box!($parse)(input)
            }
        }
    }
}

//-----------------------------------------------------------------------------
// (Helper) Color
//-----------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum Color {
    Luma(Luma<u8>),
    LumaA(LumaA<u8>),
    Rgb(Rgb<u8>),
    Rgba(Rgba<u8>),
}

fn parse_color(input: &str) -> IResult<&str, Color> {
    map(
        delimited(tag("("), nonempty_sequence(",", int::<u8>), tag(")")),
        |vs| color_from_vals(&vs)
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

//-----------------------------------------------------------------------------
// AdaptiveThreshold
//-----------------------------------------------------------------------------

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

impl_image_op_parse!(
    AdaptiveThreshold,
    "athresh <block_radius>",
    "Binarises an image using adaptive thresholding with the given block radius.",
    op_one("athresh", int::<u32>, |x| AdaptiveThreshold(x))
);

//-----------------------------------------------------------------------------
// Array
//-----------------------------------------------------------------------------

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

impl_image_op_parse!(
    Array,
    "[<image_op>, .. ]",
    "Applies a series of image operations to the stack - the nth provided operation is applied \
    to the nth item in the stack.",
    map(
        delimited(tag("["), nonempty_sequence(",", parse_image_op), tag("]")),
        |v| Array(v)
    )
);

//-----------------------------------------------------------------------------
// Blue
//-----------------------------------------------------------------------------

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

impl_image_op_parse!(
    Blue,
    "blue",
    "Extracts the blue channel from an image as a grayscale image.",
    op_zero("blue", Blue)
);

//-----------------------------------------------------------------------------
// Carve
//-----------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq)]
struct Carve(f32);

impl ImageOp for Carve {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| carve(i, self.0));
    }
}

fn carve(image: &DynamicImage, ratio: f32) -> DynamicImage {
    assert!(ratio <= 1.0);
    let target_width = (image.width() as f32 * ratio) as u32;
    dynamic_map!(image, |i| imageproc::seam_carving::shrink_width(i, target_width))
}

impl_image_op_parse!(
    Carve,
    "carve <width_ratio>",
    "Reduces the width of an image to the given multiple of its original width via seam carving.",
    op_one("carve", float, |x| Carve(x))
);

//-----------------------------------------------------------------------------
// Circle
//-----------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct Circle {
    fill: FillType,
    center: (i32, i32),
    radius: i32,
    color: Color,
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

impl_image_op_parse!(
    Circle,
    "circle <filltype> <cx> <cy> <radius> (color)",
    "Draws a circle on an image. filltype can be either 'hollow' or 'filled'. \
    color can be grayscale '(12)', grayscale with alpha '(12, 255)', RGB '(255, 0, 255)', \
    or RGBA '(128, 128, 0, 255)'.",
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
    )
);

//-----------------------------------------------------------------------------
// Const
//-----------------------------------------------------------------------------

// TODO: Over-writing with const means you have to do some awkward stack manipulation
// TODO: if you want to combine the constant image with your inputs.
// TODO: Maybe it would be better to allow constant images as 'pseudo-inputs', which get
// TODO: injected at the start of the pipeline, rather than manipulating the existing stack.

#[derive(Debug, Clone, PartialEq, Eq)]
struct Const {
    width: u32,
    height: u32,
    color: Color,
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

impl_image_op_parse!(
    Const,
    "const <width> <height> (color)",
    "Create an image with a single constant value. \
    color can be grayscale '(12)', grayscale with alpha '(12, 255)', RGB '(255, 0, 255)', \
    or RGBA '(128, 128, 0, 255)'.",
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
    )
);

//-----------------------------------------------------------------------------
// Drop
//-----------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Drop;

impl ImageOp for Drop {
    fn apply(&self, stack: &mut ImageStack) {
        stack.drop();
    }
}

impl_image_op_parse!(
    Drop,
    "DROP",
    "Discards the top element of the image stack.",
    op_zero("DROP", Drop)
);

//-----------------------------------------------------------------------------
// Dup
//-----------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Dup(usize);

impl ImageOp for Dup {
    fn apply(&self, stack: &mut ImageStack) {
        stack.dup(self.0);
    }
}

impl_image_op_parse!(
    Dup,
    "DUP( <n>)?",
    "Duplicates the top element of the image stack n times. n defaults to 1 if not provided.",
    op_one_opt("DUP", int::<usize>, |x| Dup(x.unwrap_or(1)))
);

//-----------------------------------------------------------------------------
// Func
//-----------------------------------------------------------------------------

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
    dynamic_map!(image, |i| map_subpixels_with_coords(i, f))
}

fn parse_func(input: &str) -> IResult<&str, Func> {
    let (i, (text, expr)) = crate::expr::parse_func(input, "func")?;
    Ok((i, Func { text, expr }))
}

impl_image_op_parse!(
    Func,
    "func { <fn> }",
    "Applies a user-provided function to each subpixel of an image.",
    parse_func
);

//-----------------------------------------------------------------------------
// Func2
//-----------------------------------------------------------------------------

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

fn parse_func2(input: &str) -> IResult<&str, Func2> {
    let (i, (text, expr)) = crate::expr::parse_func(input, "func2")?;
    Ok((i, Func2 { text, expr }))
}

impl_image_op_parse!(
    Func2,
    "func2 { <fn> }",
    "Applies a user-provided function to each subpixel of a pair of images.",
    parse_func2
);

//-----------------------------------------------------------------------------
// Gaussian
//-----------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq)]
struct Gaussian(f32);

impl ImageOp for Gaussian {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| gaussian(i, self.0));
    }
}

fn gaussian(image: &DynamicImage, sigma: f32) -> DynamicImage {
    dynamic_map!(image, |i| imageproc::filter::gaussian_blur_f32(i, sigma))
}

impl_image_op_parse!(
    Gaussian,
    "gaussian <standard_deviation>",
    "Applies a Gaussian blur with the specified standard deviation.",
    op_one("gaussian", float, |x| Gaussian(x))
);

//-----------------------------------------------------------------------------
// Gray
//-----------------------------------------------------------------------------

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

impl_image_op_parse!(
    Gray,
    "gray",
    "Converts an image to grayscale.",
    op_zero("gray", Gray)
);

//-----------------------------------------------------------------------------
// Green
//-----------------------------------------------------------------------------

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

impl_image_op_parse!(
    Green,
    "green",
    "Extracts the green channel from an image as a grayscale image.",
    op_zero("green", Green)
);

//-----------------------------------------------------------------------------
// Grid
//-----------------------------------------------------------------------------

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

impl_image_op_parse!(
    Grid, // TODO: roll hcat and vcat aliases into the Grid parser
    "grid <columns> <rows>",
    "Arranges images into a grid. Aliases: 'hcat' is equivalent to 'Grid 2 1', 'hcat n' \
    is equivalent to 'Grid n 1', 'vcat' is equivalent to 'Grid 1 2', 'vcat n' is equivalent \
    to 'Grid 1 n'.",
    op_two("grid", int::<u32>, int::<u32>, |w, h| Grid(w, h))
);

//-----------------------------------------------------------------------------
// HFlip
//-----------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq)]
struct HFlip;

impl ImageOp for HFlip {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| hflip(i));
    }
}

fn hflip(image: &DynamicImage) -> DynamicImage {
    dynamic_map!(image, image::imageops::flip_horizontal)
}

impl_image_op_parse!(
    HFlip,
    "hflip",
    "Flips an image horizontally.",
    op_zero("hflip", HFlip)
);

//-----------------------------------------------------------------------------
// Id
//-----------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Id;

impl ImageOp for Id {
    fn apply(&self, _: &mut ImageStack) {
        // Do nothing
    }
}

impl_image_op_parse!(
    Id,
    "id",
    "Applies the identity function, i.e. does nothing. This makes some pipelines more concise \
    to write.",
    op_zero("id", Id)
);

//-----------------------------------------------------------------------------
// Median
//-----------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Median(u32, u32);

impl ImageOp for Median {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| median(i, self.0, self.1));
    }
}

fn median(image: &DynamicImage, x_radius: u32, y_radius: u32) -> DynamicImage {
    dynamic_map!(image, |i| imageproc::filter::median_filter(i, x_radius, y_radius))
}

impl_image_op_parse!(
    Median,
    "median <x_radius> <y_radius>",
    "Applies a median filter with the given x radius and y radius.",
    op_two("median", int::<u32>, int::<u32>, |rx, ry| Median(rx, ry))
);

//-----------------------------------------------------------------------------
// OtsuThreshold
//-----------------------------------------------------------------------------

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

impl_image_op_parse!(
    OtsuThreshold,
    "othresh",
    "Binarises an image using Otsu thresholding.",
    op_zero("othresh", OtsuThreshold)
);

//-----------------------------------------------------------------------------
// Red
//-----------------------------------------------------------------------------

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

impl_image_op_parse!(
    Red,
    "red",
    "Extracts the red channel from an image as a grayscale image.",
    op_zero("red", Red)
);

//-----------------------------------------------------------------------------
// Resize
//-----------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Resize {
    width: Option<u32>,
    height: Option<u32>,
}

impl ImageOp for Resize {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| resize(i, self));
    }
}

fn resize(image: &DynamicImage, target: &Resize) -> DynamicImage {
    let (w, h) = match (target.width, target.height) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => {
            let h = ((w as f32 / image.width() as f32) * image.height() as f32) as u32;
            (w, h)
        }
        (None, Some(h)) => {
            let w = ((h as f32 / image.height() as f32) * image.width() as f32) as u32;
            (w, h)
        }
        _ => panic!("Must provide at least one of target width or target height"),
    };
    image.resize(w, h, image::imageops::FilterType::Lanczos3)
}

// see test_parse_resize for usage
fn parse_resize(input: &str) -> IResult<&str, Resize> {
    preceded(
        tag("resize"),
        alt((
            map(
                tuple((
                    preceded(space1, int::<u32>),
                    preceded(space1, int::<u32>),
                )),
                |(w, h)| Resize { width: Some(w), height: Some(h) }
            ),
            map(
                permutation((
                    preceded(space1, named_arg("w", int::<u32>)),
                    preceded(space1, named_arg("h", int::<u32>)),
                )),
                |(w, h)| Resize { width: Some(w), height: Some(h) }
            ),
            map(
                preceded(space1, named_arg("w", int::<u32>)),
                |w| Resize { width: Some(w), height: None}
            ),
            map(
                preceded(space1, named_arg("h", int::<u32>)),
                |h| Resize { width: None, height: Some(h) }
            ),
        )),
    )(input)
}

impl_image_op_parse!(
    Resize,
    "resize <width> <height>",
    "Resizes an image to the given dimensions. Also supports forms 'resize w=71', \
    'resize h=71', and 'resize w=17 h=21'. If only one of width or height is provided \
    then the target for the other dimension is chosen to preserve the image's aspect ratio.",
    parse_resize
);

//-----------------------------------------------------------------------------
// Rot
//-----------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Rot(usize);

impl ImageOp for Rot {
    fn apply(&self, stack: &mut ImageStack) {
        stack.rot(self.0);
    }
}

impl_image_op_parse!(
    Rot,
    "ROT( <n>)?",
    "Rotates the top n elements of the stack by 1. n defaults to 3 if not provided. \
    Aliases: 'SWAP' is equivalent to 'ROT 2.",
    op_one_opt("ROT", int::<usize>, |x| Rot(x.unwrap_or(3)))
);

//-----------------------------------------------------------------------------
// Rotate
//-----------------------------------------------------------------------------

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

impl_image_op_parse!(
    Rotate,
    "rotate <angle>",
    "Rotates an image clockwise about its center by the given angle in degrees.",
    op_one("rotate", float, |x| Rotate(x))
);

//-----------------------------------------------------------------------------
// Scale
//-----------------------------------------------------------------------------

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

impl_image_op_parse!(
    Scale,
    "scale <ratio>",
    "Scales image width and height by tbe given multiplier.",
    op_one("scale", float, |x| Scale(x))
);

//-----------------------------------------------------------------------------
// Sobel
//-----------------------------------------------------------------------------

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

impl_image_op_parse!(
    Sobel,
    "sobel",
    "Computes image gradients using the Sobel filter.",
    op_zero("sobel", Sobel)
);

//-----------------------------------------------------------------------------
// Translate
//-----------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Translate(i32, i32);

impl ImageOp for Translate {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| translate(i, self.0, self.1));
    }
}

fn translate(image: &DynamicImage, tx: i32, ty: i32) -> DynamicImage {
    dynamic_map!(image, |i| imageproc::geometric_transformations::translate(i, (tx, ty)))
}

impl_image_op_parse!(
    Translate,
    "translate <tx> <ty>",
    "Translates an image by (tx, ty). Positive values of tx move the image to the right, \
    and positive values of ty move it downwards.",
    op_two("translate", int::<i32>, int::<i32>, |tx, ty| Translate(tx, ty))
);

//-----------------------------------------------------------------------------
// VFlip
//-----------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq)]
struct VFlip;

impl ImageOp for VFlip {
    fn apply(&self, stack: &mut ImageStack) {
        one_in_one_out(stack, |i| vflip(i));
    }
}

fn vflip(image: &DynamicImage) -> DynamicImage {
    dynamic_map!(image, image::imageops::flip_vertical)
}

impl_image_op_parse!(
    VFlip,
    "vflip",
    "Flips an image vertically.",
    op_zero("vflip", VFlip)
);

//-----------------------------------------------------------------------------
// Mapping functions
//-----------------------------------------------------------------------------

use imageproc::{definitions::Image, map::{ChannelMap, WithChannel}};
use image::{ImageBuffer, Pixel, Primitive};

// Could add this to imageproc, but either way we should extend the syntax for custom functions
// to support user-provided functions on entire pixels rather than just per channel.
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
            for c in 0..P::CHANNEL_COUNT {
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
            for c in 0..P::CHANNEL_COUNT {
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

    // TODO: the error messages from this parser are going to be pretty useless.
    #[test]
    fn test_parse_resize() {
        // Need to provide at least one of width or height
        assert_pipeline_parse_failure("resize");
        // If providing only one parameter then need to specify
        // whether it's width or height
        assert_pipeline_parse_failure("resize 10");
        // Providing both positional arguments
        assert_pipeline_parse(
            "resize 10 12",
            &vec!["Resize { width: Some(10), height: Some(12) }"]
        );
        // Providing both named arguments
        assert_pipeline_parse(
            "resize w=10 h=12",
            &vec!["Resize { width: Some(10), height: Some(12) }"]
        );
        // Providing both named arguments in opposite order
        assert_pipeline_parse(
            "resize h=12 w=10",
            &vec!["Resize { width: Some(10), height: Some(12) }"]
        );
        // Width only
        assert_pipeline_parse(
            "resize w=10",
            &vec!["Resize { width: Some(10), height: None }"]
        );
        // Height only
        assert_pipeline_parse(
            "resize h=12",
            &vec!["Resize { width: None, height: Some(12) }"]
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
            "gray > func { 255 * (p > 100) } > rotate 45 > othresh > scale 2 > resize w=7"
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

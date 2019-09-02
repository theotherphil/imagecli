use std::path::PathBuf;
use structopt::StructOpt;
use image::{DynamicImage, GenericImageView, open};

mod stack;
mod image_ops;
use crate::image_ops::ImageOp;
mod stack_ops;
use crate::stack_ops::StackOp;
mod error;
use crate::error::Result;

type ImageStack = stack::Stack<DynamicImage>;

// TODO:
// - mechanism for taking an array of functions Image^n_i -> Image^m_i and producing a single
//   function Image^{\Sigma n_i} -> Image^{\Sigma m_i}
//  
//  e.g. 2 x 2 grid of blurred images becomes:
//      ... 'scale 0.4 > DUP 3 > [id, gaussian 5.0, gaussian 10.0, gaussian 15.0] > grid 2 2'
//
// - write grid properly, and either remove hcat and vcat or just make them shorthand for grid 2 1 and grid 1 2
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
//  - add constant images
//  - add ability to add margins
//  - support user-defined functions over pixels, e.g. map \x -> x * 0.3
//  - add support for everything else relevant from imageproc
//  - add validation for parsing of individual stages (make them subcommands somehow?)
//      or use a parser combinatory library, e.g. nom
//  - add specific subcommands for the pipeline stages?
//      e.g. make -p 'scale 0.4' equivalent to --scale 0.4
//  - add stages with optional parameters. e.g. rotate uses image centre by default, but user
//      can specify a centre of rotation if they like

#[derive(StructOpt, Debug)]
struct Opt {
    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: u8,

    /// Input files
    #[structopt(short, long, parse(from_os_str))]
    input: Vec<PathBuf>,

    /// Output files
    #[structopt(short, long, parse(from_os_str))]
    output: Vec<PathBuf>,

    /// Image processing pipeline to apply
    #[structopt(short, long)]
    pipeline: Option<String>,
}

fn main() -> Result<()> {
    let opt = Opt::from_args();
    let verbose = opt.verbose > 0;

    let inputs: Vec<(&PathBuf, DynamicImage)> = opt.input
        .iter()
        .map(|p| (p, open(p).unwrap()))
        .collect();

    if verbose {
        for (path, image) in &inputs {
            println!("Input image {:?} (width: {}, height: {})", path, image.width(), image.height());
        }
    }
    let outputs = if opt.pipeline.is_some() {
        let inputs = inputs.into_iter().map(|(_, i)| i).collect(); 
        run_pipeline(&opt.pipeline.unwrap(), inputs, opt.verbose > 0)
    } else {
        inputs.into_iter().map(|(_, i)| i).collect()
    };
    for (path, image) in opt.output.iter().zip(outputs) {
        if verbose {
            println!("Output image {:?} (width: {}, height: {})", path, image.width(), image.height());
        }
        image.save(path)?;
    }
    Ok(())
}

fn run_pipeline(pipeline: &str, inputs: Vec<DynamicImage>, verbose: bool) -> Vec<DynamicImage> {
    // TODO: validation!
    let ops: Vec<Op> = pipeline.split('>').map(|s| s.trim()).map(|s| Op::parse(s).unwrap()).collect();
    let mut stack = ImageStack::new(inputs);

    for op in ops {
        if verbose {
            println!("Applying {:?}", op);
        }
        op.apply(&mut stack);
    }

    stack.contents()
}

/// A pipeline operation - either a direct manipulation of the stack,
/// or an image operation which reads from and writes to the top of the stack.
#[derive(Debug)]
pub enum Op {
    StackOp(StackOp),
    ImageOp(Box<dyn ImageOp>),
}

impl Op {
    fn parse(op: &str) -> Option<Op> {
        match StackOp::parse(op) {
            Some(o) => Some(Op::StackOp(o)),
            None => match image_ops::parse(op) {
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

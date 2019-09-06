use std::path::PathBuf;
use structopt::StructOpt;
use image::{DynamicImage, GenericImageView, open};

mod stack;
mod image_ops;
use crate::image_ops::{ImageOp, parse};
mod expr;
mod error;
use crate::error::Result;

type ImageStack = stack::Stack<DynamicImage>;

// TODO:
//  - add tests for parsing ops
//  - support different signatures in Array op
//  - example showing changing format of all images in a directory maching some pattern
//  - flip, thumbnails, crop, allow targeting a fixed size image
//  - add ability to add margins
//  - add validation for parsing of individual stages (make them subcommands somehow?)
//      or use a parser combinatory library, e.g. nom
//  - add specific subcommands for the pipeline stages?
//      e.g. make -p 'scale 0.4' equivalent to --scale 0.4
//  - add stages with optional parameters. e.g. rotate uses image centre by default, but user
//      can specify a centre of rotation if they like
//  - make performance less atrocious. Don't just clone everything all the time

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
    let ops: Vec<Box<dyn ImageOp>> = pipeline
        .split('>')
        .map(|s| s.trim())
        .map(|s| parse(s).unwrap())
        .collect();

    let mut stack = ImageStack::new(inputs);

    for op in ops {
        if verbose {
            println!("Applying {:?}", op);
        }
        op.apply(&mut stack);
    }

    stack.contents()
}

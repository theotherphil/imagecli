#![cfg_attr(test, feature(test))]

use std::path::PathBuf;
use structopt::StructOpt;
use image::{DynamicImage, GenericImageView, open};

mod stack;
mod image_ops;
use crate::image_ops::{run_pipeline, documentation};
mod expr;
mod error;
use crate::error::Result;
mod parse_utils;

#[cfg(test)]
extern crate test;

type ImageStack = stack::Stack<DynamicImage>;

// TODO:
//  - write user guide
//  - convert the readme examples into tests
//  - support different signatures in Array op
//  - scale is extremely slow. Fix this.
//  - example showing changing format of all images in a directory matching some pattern
//  - image::imageops::crop(img, x, y, w, h),
//  - image::imageops::overlay(bottom, top, x, y),
//  - add ability to add margins - need to be able to specify left, right, top, bottom
//  - make performance less atrocious. Don't just clone everything all the time
//  - add mask operation, and example of masking with a circle
//  - add ability to handle entire pixels in user-defined funcs, not just subpixels.
//    access via p.0, p.1, etc.? Write example rotating red green and blue channels
//    rgb(p.1, p.2, p.0)
//  - think about adding support for named parameters. check effect on perf (and compile times...)
//  - support passing multiple images to a single pipeline or running an independent pipeline on each
//    image. Add glob support for inputs (and make listing outputs work with this)

#[derive(StructOpt, Debug)]
struct Opt {
    /// Verbose mode (-v, -vv, -vvv, etc.).
    #[structopt(short, long, parse(from_occurrences))]
    verbose: u8,

    /// Input files.
    #[structopt(short, long, parse(from_os_str))]
    input: Vec<PathBuf>,

    /// Output files.
    #[structopt(short, long, parse(from_os_str))]
    output: Vec<PathBuf>,

    /// Image processing pipeline to apply.
    #[structopt(short, long)]
    pipeline: Option<String>,

    /// Print documentation for the available pipeline operations.
    #[structopt(long)]
    print_ops: bool
}

fn main() -> Result<()> {
    let opt = Opt::from_args();

    if opt.print_ops {
        for docs in documentation().iter() {
            println!( "---------------------------------");
            println!( "Operation: {}", docs.operation);
            println!( "Usage: {}", docs.usage);
            println!( "---------------------------------");
            println!( "{}", docs.explanation);
        }
    }

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

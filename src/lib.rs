#![cfg_attr(test, feature(test))]
#![allow(
    clippy::many_single_char_names,
    clippy::single_match,
    clippy::float_cmp,
    clippy::cast_lossless
)]

use image::{open, DynamicImage, GenericImageView};
use std::path::PathBuf;

pub mod documentation;
pub mod error;
use crate::error::Result;
mod example;
mod expr;
pub mod image_ops;
use image_ops::parse;
mod parse_utils;
mod stack;

#[cfg(test)]
extern crate test;

pub type ImageStack = stack::Stack<DynamicImage>;

/// Load inputs, run the pipeline, and save the results.
pub fn process(
    input_paths: &[PathBuf],
    output_paths: &[PathBuf],
    pipeline: Option<String>,
    verbose: bool,
) -> Result<()> {
    // Load inputs
    let mut inputs = Vec::new();
    for path in input_paths.iter() {
        let image = open(path)?;
        if verbose {
            println!(
                "Loaded input {:?} (width: {}, height: {})",
                path,
                image.width(),
                image.height()
            );
        }
        inputs.push(image);
    }

    // Run pipeline
    let pipeline = match pipeline {
        Some(p) => p.clone(),
        None => "".into(),
    };
    let outputs = run_pipeline(&pipeline, inputs, verbose);

    // Save results
    for (path, image) in output_paths.iter().zip(outputs) {
        if verbose {
            println!(
                "Output image {:?} (width: {}, height: {})",
                path,
                image.width(),
                image.height()
            );
        }
        image.save(path)?;
    }

    Ok(())
}

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

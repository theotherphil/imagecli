//! A command line image processing tool, built on top of
//! the [image] and [imageproc] crates.
//!
//! [image]: https://github.com/image-rs/image
//! [imageproc]: https://github.com/image-rs/imageproc

#![deny(missing_docs)]
#![cfg_attr(test, feature(test))]
#![allow(
    clippy::many_single_char_names,
    clippy::single_match,
    clippy::float_cmp,
    clippy::cast_lossless
)]

use image::{open, DynamicImage, GenericImageView};
use snafu::ResultExt;
use std::path::PathBuf;

pub mod documentation;
pub mod error;
use crate::error::{ImageOpenError, ImageSaveError, Result};
mod example;
mod expr;
pub mod image_ops;
use image_ops::parse;
mod parse_utils;
mod stack;

#[cfg(test)]
extern crate test;

/// An image stack. All image operations in this library
/// operate on an image stack by popping zero or more images
/// from it, computing zero or more results and pushing those
/// back onto the stack.
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
        let image = open(path).context(ImageOpenError { path })?;
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
    let outputs = run_pipeline(&pipeline, inputs, verbose)?;

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
        image.save(path).context(ImageSaveError { path })?;
    }

    Ok(())
}

/// Run an image processing pipeline on a stack with the given initial contents.
pub fn run_pipeline(
    pipeline: &str,
    inputs: Vec<DynamicImage>,
    verbose: bool,
) -> Result<Vec<DynamicImage>> {
    let mut stack = ImageStack::new(inputs);
    let ops = parse(pipeline)?;

    for op in ops {
        if verbose {
            println!("Applying {:?}", op);
        }
        op.apply(&mut stack);
    }

    Ok(stack.contents())
}

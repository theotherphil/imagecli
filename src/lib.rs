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

pub mod documentation;
pub mod error;
use crate::error::{
    GlobIterationError, GlobPatternError, ImageCliError, ImageOpenError, ImageSaveError, Result,
};
mod example;
mod expr;
use glob::glob;
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
    input_patterns: &[String],
    output_patterns: &[String],
    pipeline: Option<String>,
    verbose: bool,
) -> Result<()> {
    // Load inputs
    let mut inputs = Vec::new();

    let paths = input_patterns
        .iter()
        .map(|pattern| glob(pattern).context(GlobPatternError { pattern }))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .map(|path| path.context(GlobIterationError))
        .collect::<Result<Vec<_>>>()?;

    for path in paths {
        let path = &path;
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
    let using_index_var = output_patterns.iter().any(|p| p.contains("$N"));
    if using_index_var {
        if output_patterns.len() != 1 {
            return Err(ImageCliError::InvalidArgError {
                context: "If --outputs contains a path containing $N then it cannot \
                          contain any other paths."
                    .into(),
            });
        }
        let output_pattern = &output_patterns[0];
        for (index, image) in outputs.iter().enumerate() {
            println!("INDEX: {}", index);
            let path = output_pattern.replace("$N", &index.to_string());
            let path = &path;
            println!("PATH: {}", path);
            if verbose {
                log_output(image, path);
            }
            image.save(path).context(ImageSaveError { path })?;
        }
    } else {
        for (path, image) in output_patterns.iter().zip(outputs) {
            if verbose {
                log_output(&image, path);
            }
            image.save(path).context(ImageSaveError { path })?;
        }
    }

    Ok(())
}

fn log_output(image: &DynamicImage, path: &str) {
    println!(
        "Output image {:?} (width: {}, height: {})",
        path,
        image.width(),
        image.height()
    );
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

//! A command line image processing tool, built on top of
//! the [image] and [imageproc] crates.
//!
//! [image]: https://github.com/image-rs/image
//! [imageproc]: https://github.com/image-rs/imageproc

use image::{open, DynamicImage, GenericImageView};
use snafu::ResultExt;
use std::{
    io::{Error, ErrorKind},
    path::PathBuf,
};

pub mod documentation;
pub mod error;
use crate::error::{
    GlobIterationError, GlobPatternError, ImageCliError, ImageOpenError, ImageSaveError, Result,
};
mod example;
mod expr;
use glob::glob;
pub mod image_ops;
use image_ops::{parse, ImageOp};
mod output_spec;
use output_spec::OutputSpec;
mod parse_utils;
mod stack;

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
    let inputs = load_inputs(input_patterns, verbose)?;
    let output_spec = OutputSpec::parse(output_patterns)?;

    let pipeline = match pipeline {
        Some(p) => parse(&p)?,
        None => vec![],
    };
    let outputs = run_pipeline(&pipeline, inputs, verbose)?;
    save_images(&output_spec, &outputs, verbose)?;

    Ok(())
}

/// Run an image processing pipeline on a stack with the given initial contents.
pub fn run_pipeline(
    pipeline: &[Box<dyn ImageOp>],
    inputs: Vec<DynamicImage>,
    verbose: bool,
) -> Result<Vec<DynamicImage>> {
    let mut stack = ImageStack::new(inputs);
    for op in pipeline {
        if verbose {
            println!("Applying {:?}", op);
        }
        op.apply(&mut stack);
    }
    Ok(stack.contents())
}

/// Finds the set of input paths matching the provided glob patterns.
pub fn input_paths(input_patterns: &[String]) -> Result<Vec<PathBuf>> {
    input_patterns
        .iter()
        .map(|pattern| paths_matching_pattern(pattern))
        .collect::<Result<Vec<_>>>()
        .map(|v| v.into_iter().flatten().collect())
}

/// Finds the set of paths matching the provided glob pattern.
pub fn paths_matching_pattern(pattern: &str) -> Result<Vec<PathBuf>> {
    let glob = glob(pattern).context(GlobPatternError { pattern })?;
    let paths = glob
        .map(|p| p.context(GlobIterationError))
        .collect::<Result<Vec<_>>>();

    // A bit of a hack for https://github.com/theotherphil/imagecli/issues/42
    if let Ok(p) = &paths {
        if p.len() == 0 && !pattern.contains("*") && !pattern.contains("?") {
            return Err(ImageCliError::IoError {
                context: "Input error".into(),
                source: Error::new(
                    ErrorKind::NotFound,
                    format!("No file found matching pattern '{}'", pattern),
                ),
            });
        }
    }

    paths
}

/// Load all images matching the given globs.
pub fn load_inputs(input_patterns: &[String], verbose: bool) -> Result<Vec<DynamicImage>> {
    let mut inputs = Vec::new();

    for pattern in input_patterns {
        let paths = paths_matching_pattern(pattern)?;
        if verbose {
            println!(
                "Found {} path(s) matching input pattern {}: {:?}",
                paths.len(),
                pattern,
                paths,
            );
        }

        for path in &paths {
            let image = open(&path).context(ImageOpenError { path: &path })?;
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
    }

    Ok(inputs)
}

/// Save images according to an `OutputSpec`.
pub fn save_images(spec: &OutputSpec, images: &[DynamicImage], verbose: bool) -> Result<()> {
    for (index, image) in images.iter().enumerate() {
        match spec.nth_output_path(index) {
            Some(path) => {
                if verbose {
                    log_output(&image, &path);
                }
                image.save(&path).context(ImageSaveError { path: &path })?;
            }
            None => break,
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

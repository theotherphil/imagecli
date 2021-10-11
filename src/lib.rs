//! A command line image processing tool, built on top of
//! the [image] and [imageproc] crates.
//!
//! [image]: https://github.com/image-rs/image
//! [imageproc]: https://github.com/image-rs/imageproc

use image::{open, DynamicImage, GenericImageView};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use snafu::ResultExt;
use std::{
    io::{Error, ErrorKind},
    ops::AddAssign,
    path::PathBuf,
    sync::{Arc, Mutex},
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
    pipeline_string: Option<String>,
    verbose: bool,
    parallel: bool,
) -> Result<()> {
    let input_filenames = load_inputs_filename(input_patterns, verbose)?;
    let pipeline = match &pipeline_string {
        Some(p) => parse(p)?,
        None => vec![],
    };
    let output_spec = OutputSpec::parse(output_patterns)?;
    if parallel {
        if let Some(required_images) = total_operation_input_images(&pipeline) {
            let images_counter = Arc::new(Mutex::new(0));
            input_filenames
                .par_chunks_exact(required_images)
                .map(|input_filename_bucket| {
                    let pipeline = match &pipeline_string {
                        Some(p) => parse(p)?,
                        None => vec![],
                    };
                    process_filenames(
                        &input_filename_bucket,
                        &output_spec,
                        &pipeline,
                        verbose,
                        images_counter.clone(),
                    )
                })
                .collect::<Result<Vec<()>>>()?;
            Ok(())
        } else {
            eprintln!("Cannot determine if its possible to parallelize the given sequence of operations. Please omit the flag.");
            Ok(())
        }
    } else {
        process_filenames(
            &input_filenames,
            &output_spec,
            &pipeline,
            verbose,
            Arc::new(Mutex::new(0)),
        )?;
        Ok(())
    }
}

/// Load inputs, run the pipeline, and save the results.
pub fn process_filenames(
    input_filenames: &[PathBuf],
    output_spec: &OutputSpec,
    pipeline: &[Box<dyn ImageOp>],
    verbose: bool,
    images_counter: Arc<Mutex<usize>>,
) -> Result<()> {
    let input_images = load_inputs_image(input_filenames, verbose)?;
    let outputs = run_pipeline(&pipeline, input_images, verbose)?;
    save_images(&output_spec, &outputs, verbose, images_counter)?;
    Ok(())
}

/// Determines the total input/output signature of a given sequence of image operations
pub fn total_operation_input_images(pipeline: &[Box<dyn ImageOp>]) -> Option<usize> {
    let total = pipeline
        .iter()
        .map(|s| {
            s.signature()
                .map(|s| std::iter::once(-(s.0 as i32)).chain(std::iter::once(s.1 as i32)))
        })
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .flatten()
        .scan(0, |a, b| {
            *a = *a + b;
            Some(b)
        })
        .min();
    if let Some(total) = total {
        if total <= 0 {
            Some((-total) as usize)
        } else {
            None
        }
    } else {
        None
    }
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
pub fn load_inputs_image(input_patterns: &[PathBuf], verbose: bool) -> Result<Vec<DynamicImage>> {
    let mut inputs = Vec::with_capacity(input_patterns.len());
    for path in input_patterns {
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
    Ok(inputs)
}

/// Load all images matching the given globs.
pub fn load_inputs_filename(input_patterns: &[String], verbose: bool) -> Result<Vec<PathBuf>> {
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
        inputs.extend(paths);
    }

    Ok(inputs)
}

/// Save images according to an `OutputSpec`.
pub fn save_images(
    spec: &OutputSpec,
    images: &[DynamicImage],
    verbose: bool,
    images_counter: Arc<Mutex<usize>>,
) -> Result<()> {
    let image_counter = {
        let mut counter = images_counter.lock().unwrap();
        let counter_copy = counter.clone();
        counter.add_assign(images.len());
        counter_copy
    };
    for (index, image) in images.iter().enumerate() {
        match spec.nth_output_path(image_counter + index) {
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

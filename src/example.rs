//! Types representing example pipelines to run, and functions to run them.

use crate::{
    error::{ImageOpenError, IoError, Result},
    input_paths,
    output_spec::OutputSpec,
    run_pipeline, save_images,
};
use image::open;
use snafu::ResultExt;
use std::fmt::Write;
use std::path::Path;

/// The default image file names to use as inputs for example pipelines.
/// Pipelines with n inputs consume the first n images in this array.
const DEFAULT_INPUTS: &[&str] = &["robin.png", "robin_gray.png", "yellow.png"];

/// An example pipeline to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Example {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub pipeline: String,
    /// The files within /images to use as inputs. Glob patterns are supported.
    /// If not specified then the first num_inputs names from DEFAULT_EXAMPLE_INPUTS are used.
    pub override_input_names: Option<Vec<String>>,
    /// By default all output files have a png file extension.
    pub output_file_extension: Option<String>,
    /// If false then we create a new named output path for each image.
    /// If true then we use a single output path containing `{n}`.
    /// (But we still require `num_outputs` to be set in order to generate
    /// correct documentation.)
    pub variable_outputs: bool,
    pub style: Style,
}

/// Whether to render an example command line with long or short
/// forms of parameters (e.g. --input or -i).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Style {
    LongForm,
    ShortForm,
}

impl Example {
    pub fn new(num_inputs: usize, num_outputs: usize, pipeline: &'static str) -> Example {
        Example {
            num_inputs,
            num_outputs,
            pipeline: pipeline.into(),
            override_input_names: None,
            output_file_extension: None,
            variable_outputs: false,
            style: Style::ShortForm,
        }
    }

    pub fn instantiate(
        &self,
        input_dir: String,
        output_dir: String,
        output_root_name: String,
    ) -> InstantiatedExample {
        let input_file_names = match &self.override_input_names {
            Some(n) => n.clone(),
            None => DEFAULT_INPUTS[..self.num_inputs]
                .iter()
                .map(|f| f.to_string())
                .collect(),
        };

        let ext = self
            .output_file_extension
            .as_ref()
            .map(|e| e.as_str())
            .unwrap_or("png");

        let output_file_names = if self.variable_outputs {
            vec![format!("{}_{{n}}.{}", output_root_name, ext)]
        } else {
            (0..self.num_outputs)
                .map(|c| format!("{}_{}.{}", output_root_name, c, ext))
                .collect()
        };

        InstantiatedExample {
            input_dir,
            input_file_names,
            output_dir,
            output_file_names,
            pipeline: self.pipeline.clone(),
            style: self.style,
            variable_outputs: self.variable_outputs,
            num_outputs: self.num_outputs,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstantiatedExample {
    pub input_dir: String,
    pub input_file_names: Vec<String>,
    pub output_dir: String,
    pub output_file_names: Vec<String>,
    pub pipeline: String,
    pub style: Style,
    pub variable_outputs: bool,
    pub num_outputs: usize,
}

impl InstantiatedExample {
    /// Generates markdown containing command line and links to output files
    /// to include in documentation.
    pub fn render_for_documentation(&self) -> String {
        let mut result = String::new();
        writeln!(
            result,
            "<pre>{}</pre>",
            self.command_line_for_documentation()
        )
        .unwrap();
        if self.variable_outputs {
            for i in 0..self.num_outputs {
                writeln!(
                    result,
                    "<img src='{}'/>",
                    self.output_paths()[0].replace("{n}", &i.to_string())
                )
                .unwrap();
            }
        } else {
            for link in &self.output_paths() {
                writeln!(result, "<img src='{}'/>", link).unwrap();
            }
        }
        result
    }

    /// Run this example and write output files.
    pub fn run(&self, verbose: bool) -> Result<()> {
        let input_patterns: Vec<_> = self
            .input_file_names
            .iter()
            .map(|f| Path::new(&self.input_dir).join(f).display().to_string())
            .collect();

        let input_paths = input_paths(&input_patterns)?;

        if verbose {
            println!("Input paths:\n\t{:?}", input_paths);
        }

        let inputs = input_paths
            .iter()
            .map(|p| open(&p).context(ImageOpenError { path: &p }))
            .collect::<Result<Vec<_>>>()?;

        let output_paths: Vec<_> = self
            .output_file_names
            .iter()
            .map(|f| Path::new(&self.output_dir).join(f).display().to_string())
            .collect();

        if verbose {
            println!("Output paths:\n\t{:?}", output_paths);
        }

        let output_spec = OutputSpec::parse(&output_paths)?;
        let outputs = run_pipeline(&self.pipeline, inputs, verbose)?;

        if !Path::new(&self.output_dir).is_dir() {
            std::fs::create_dir(&self.output_dir).context(IoError {
                context: format!("Unable to create directory '{}", &self.output_dir),
            })?;
        }

        save_images(&output_spec, &outputs, verbose)?;

        Ok(())
    }

    /// Produce a command line to include in documentation.
    pub fn command_line_for_documentation(&self) -> String {
        let (input_opt, output_opt, pipeline_opt) = match self.style {
            Style::LongForm => ("--input", "--output", "--pipeline"),
            Style::ShortForm => ("-i", "-o", "-p"),
        };
        let input = if !self.input_file_names.is_empty() {
            format!(" {} {}", input_opt, self.input_file_names.join(" "))
        } else {
            String::new()
        };
        let pipeline = if !self.pipeline.is_empty() {
            format!(" {} '{}'", pipeline_opt, self.pipeline)
        } else {
            String::new()
        };
        format!(
            "imagecli{} {} {}{}",
            input,
            output_opt,
            self.output_file_names.join(" "),
            pipeline,
        )
    }

    fn output_paths(&self) -> Vec<String> {
        self.output_file_names
            .iter()
            .map(|f| Path::new(&self.output_dir).join(f).display().to_string())
            .collect()
    }
}

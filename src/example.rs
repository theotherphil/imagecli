//! Types representing example pipelines to run, and functions to run them.

use std::path::Path;
use std::fmt::Write;
use image::open;
use crate::run_pipeline;

/// The default image file names to use as inputs for example pipelines.
/// Pipelines with n inputs consume the first n images in this array.
const DEFAULT_INPUTS: &[&str] = &[
    "robin.png",
    "robin_gray.png",
    "yellow.png",
];

/// An example pipeline to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Example {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub pipeline: String,
    /// The files within /images to use as inputs. If not specified then
    /// the first num_inputs names from DEFAULT_EXAMPLE_INPUTS are used.
    pub override_input_names: Option<Vec<String>>,
    /// By default all output files have a png file extension.
    pub output_file_extension: Option<String>,
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

        let ext = self.output_file_extension.as_ref().map(|e| e.as_str()).unwrap_or("png");
        let output_file_names =  (0..self.num_outputs)
            .map(|c| format!("{}_{}.{}", output_root_name, c, ext))
            .collect();

        InstantiatedExample {
            input_dir,
            input_file_names,
            output_dir,
            output_file_names,
            pipeline: self.pipeline.clone(),
            style: self.style,
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
        ).unwrap();
        for link in &self.output_paths() {
            writeln!(result, "<img src='{}'/>", link).unwrap();
        }
        result
    }

    /// Run this example and write output files.
    pub fn run(&self) {
        let input_paths = self
            .input_file_names
            .iter()
            .map(|f| Path::new(&self.input_dir).join(f));

        let inputs = input_paths.map(|p| open(p).unwrap()).collect();

        let output_paths: Vec<_> = self
            .output_file_names
            .iter()
            .map(|f| Path::new(&self.output_dir).join(f))
            .collect();

        let outputs = run_pipeline(&self.pipeline, inputs, false);

        if !Path::new(&self.output_dir).is_dir() {
            std::fs::create_dir(&self.output_dir).unwrap();
        }

        for (path, image) in output_paths.iter().zip(outputs) {
            image.save(path).unwrap();
        }
    }

    fn command_line_for_documentation(&self) -> String {
        let (input_opt, output_opt, pipeline_opt) = match self.style {
            Style::LongForm => ("--input", "--output", "--pipeline"),
            Style::ShortForm => ("-i", "-o", "-p"),
        };
        let pipeline = if !self.pipeline.is_empty() {
            format!(" {} '{}'", pipeline_opt, self.pipeline)
        } else {
            String::new()
        };
        format!(
            "imagecli {} {} {} {}{}",
            input_opt,
            self.input_file_names.join(" "),
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


use crate::error::{ImageCliError, Result};

/// Defines the paths to use when saving output images.
pub enum OutputSpec {
    /// A list of outputs paths. These are zipped with the final contents
    /// of the image stack. Overhangs in either direction (more images than paths,
    /// or vice versa) are silently dropped.
    Paths(Vec<String>),
    /// A pattern determining where the `n`th output image will go.
    /// The second argument is the number of digits to use when writing `n` to the output path.
    Pattern(String, usize),
}

impl OutputSpec {
    /// Parses the user-provided list of output paths.
    ///
    /// This must either be a single output containing `$N` as an infix,
    /// or any number of output paths which do not contain `$N`.
    pub fn parse(paths: &[String]) -> Result<OutputSpec> {
        let spec = if paths.iter().any(|p| p.contains("$N")) {
            if paths.len() != 1 {
                let err = "If --outputs contains a path containing $N then it cannot contain \
                           any other paths.";
                return Err(ImageCliError::InvalidArgError {
                    context: err.into(),
                });
            }
            OutputSpec::Pattern(paths[0].to_string(), 3)
        } else {
            OutputSpec::Paths(paths.iter().cloned().collect())
        };
        Ok(spec)
    }

    /// Returns the path to use when saving the `n`th image in the stack.
    pub fn nth_output_path(&self, n: usize) -> Option<String> {
        match self {
            OutputSpec::Paths(paths) => {
                paths.get(n).map(|p| p.to_string())
            },
            OutputSpec::Pattern(pattern, _width) => {
                Some(pattern.replace("$N", &n.to_string()))
            }
        }
    }
}

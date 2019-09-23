use crate::error::{ImageCliError, Result};
use lazy_static::lazy_static;
use regex::Regex;

/// Defines the paths to use when saving output images.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputSpec {
    /// A list of outputs paths. These are zipped with the final contents
    /// of the image stack. Overhangs in either direction (more images than paths,
    /// or vice versa) are silently dropped.
    Paths(Vec<String>),
    /// A pattern determining where the `n`th output image will go.
    /// The second argument is the number of digits to use when writing `n` to the output path.
    Pattern(String, Option<usize>),
}

lazy_static! {
    static ref OUTPUT_PATTERN: Regex = Regex::new(r"[{{]n(:(\d+))?[}}]").unwrap();
}

fn arg_error<I: Into<String>>(msg: I) -> ImageCliError {
    ImageCliError::InvalidArgError {
        context: msg.into(),
    }
}

impl OutputSpec {
    /// Parses the user-provided list of output paths.
    ///
    /// This must either be a single output containing `{n}` or `{n:WIDTH}` as an infix,
    /// or any number of output paths which do not contain such an infix.
    pub fn parse(paths: &[String]) -> Result<OutputSpec> {
        let patterns = paths
            .iter()
            .map(|p| parse_pattern(p))
            .collect::<Result<Vec<_>>>()?;

        let patterns: Vec<_> = patterns.iter().filter_map(|p| p.as_ref()).collect();

        let spec = if !patterns.is_empty() {
            if paths.len() != 1 {
                return Err(arg_error(
                    "If --outputs contains a path containing '{n}' or '{n:WIDTH}' \
                     then it cannot contain any other paths.",
                ));
            }
            OutputSpec::Pattern(patterns[0].0.clone(), patterns[0].1)
        } else {
            OutputSpec::Paths(paths.to_vec())
        };
        Ok(spec)
    }

    /// Returns the path to use when saving the `n`th image in the stack.
    pub fn nth_output_path(&self, n: usize) -> Option<String> {
        match self {
            OutputSpec::Paths(paths) => paths.get(n).map(|p| p.to_string()),
            OutputSpec::Pattern(pattern, width) => Some(
                OUTPUT_PATTERN
                    .replace(pattern, |_caps: &regex::Captures| {
                        format!("{:0width$}", n, width = width.unwrap_or(1))
                    })
                    .to_string(),
            ),
        }
    }
}

fn parse_pattern(path: &str) -> Result<Option<(String, Option<usize>)>> {
    let captures: Vec<_> = OUTPUT_PATTERN.captures_iter(path).collect();
    match captures.len() {
        0 => Ok(None),
        1 => {
            let pat = &captures[0];
            let width = pat
                .get(2)
                .map(|w| w.as_str().parse::<usize>())
                .transpose()
                .map_err(|err| {
                    arg_error(format!(
                        "Unable to parse '{:?}' as an output pattern: {:?}",
                        pat, err
                    ))
                })?;

            Ok(Some((path.into(), width)))
        }
        _ => Err(arg_error(
            "'{n:WIDTH}' or '{n}' can only appear once in an output path",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_spec_parse_without_patterns() {
        let paths = vec!["foo.png".into(), "bar.png".into()];
        let spec = OutputSpec::parse(&paths);
        assert!(spec.is_ok());
        let spec = spec.unwrap();
        assert_eq!(spec, OutputSpec::Paths(paths.to_vec()));
        assert_eq!(spec.nth_output_path(0), Some("foo.png".into()),);
        assert_eq!(spec.nth_output_path(1), Some("bar.png".into()),);
        assert_eq!(spec.nth_output_path(2), None,);
    }

    #[test]
    fn output_spec_pattern_with_other_paths() {
        let paths = vec!["foo.png".into(), "bar{n}.png".into()];
        let spec = OutputSpec::parse(&paths);
        assert!(spec.is_err());
    }

    #[test]
    fn output_spec_pattern_without_width() {
        let paths = vec!["foo{n}.png".into()];
        let spec = OutputSpec::parse(&paths);
        assert!(spec.is_ok());
        let spec = spec.unwrap();
        assert_eq!(spec, OutputSpec::Pattern("foo{n}.png".into(), None),);
        assert_eq!(spec.nth_output_path(0), Some("foo0.png".into()),);
        assert_eq!(spec.nth_output_path(1), Some("foo1.png".into()),);
    }

    #[test]
    fn output_spec_pattern_with_width() {
        let paths = vec!["foo{n:2}.png".into()];
        let spec = OutputSpec::parse(&paths);
        assert!(spec.is_ok());
        let spec = spec.unwrap();
        assert_eq!(spec, OutputSpec::Pattern("foo{n:2}.png".into(), Some(2)),);
        assert_eq!(spec.nth_output_path(0), Some("foo00.png".into()),);
        assert_eq!(spec.nth_output_path(1), Some("foo01.png".into()),);
        assert_eq!(spec.nth_output_path(100), Some("foo100.png".into()),);
    }
}

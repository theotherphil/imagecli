//! The standard error and result types used in this library.

use snafu::Snafu;
use std::path::PathBuf;

/// The standard result type used in this library.
pub type Result<T> = std::result::Result<T, ImageCliError>;

/// The standard error type used in this library.
#[allow(missing_docs)]
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum ImageCliError {
    /// An error when attempting to open an image file.
    #[snafu(display("Unable to open image '{}': {}", path.display(), source))]
    ImageOpenError {
        path: PathBuf,
        source: image::ImageError,
    },

    /// An error when attempting to save an image.
    #[snafu(display("Unable to save image to '{}': {}", path.display(), source))]
    ImageSaveError {
        path: PathBuf,
        source: std::io::Error,
    },

    /// A generic IO error, with an ad-hoc context.
    #[snafu(display("{}: {}", context, source))]
    IoError {
        context: String,
        source: std::io::Error,
    },

    /// A generic Fmt error.
    FmtError { source: std::fmt::Error },

    /// An error when attempting to parse a pipeline.
    #[snafu(display(
        "Unable to parse pipeline.\n\nConsumed: '{}'\nRemaining: '{}'\n\nThe error is likely near the start of the remaining (unparsed) text.",
        consumed,
        remaining)
    )]
    PipelineParseError { consumed: String, remaining: String },

    /// An error from any other issue with user-provided arguments.
    #[snafu(display("{}", context))]
    InvalidArgError {
        context: String,
    },
}

impl From<std::fmt::Error> for ImageCliError {
    fn from(error: std::fmt::Error) -> Self {
        ImageCliError::FmtError { source: error }
    }
}

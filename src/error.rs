//! The standard error and result types used in this library.

/// The standard result type used in this library.
pub type Result<T> = std::result::Result<T, ImageCliError>;

/// A trivial wrapper for an Io, Fmt or image error.
#[derive(Debug)]
pub enum ImageCliError {
    /// An image error.
    ImageError(image::ImageError),
    /// An Io error.
    IoError(std::io::Error),
    /// An error writing to a formatter.
    FmtError(std::fmt::Error),
}

impl From<image::ImageError> for ImageCliError {
    fn from(error: image::ImageError) -> Self {
        ImageCliError::ImageError(error)
    }
}

impl From<std::io::Error> for ImageCliError {
    fn from(error: std::io::Error) -> Self {
        ImageCliError::IoError(error)
    }
}

impl From<std::fmt::Error> for ImageCliError {
    fn from(error: std::fmt::Error) -> Self {
        ImageCliError::FmtError(error)
    }
}

impl std::fmt::Display for ImageCliError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ImageCliError::ImageError(e) => e.fmt(f),
            ImageCliError::IoError(e) => e.fmt(f),
            ImageCliError::FmtError(e) => e.fmt(f),
        }
    }
}

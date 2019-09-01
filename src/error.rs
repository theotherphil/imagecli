
pub type Result<T> = std::result::Result<T, ImageCliError>;

#[derive(Debug)]
pub enum ImageCliError {
    ImageError(image::ImageError),
    IoError(std::io::Error),
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

impl std::fmt::Display for ImageCliError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ImageCliError::ImageError(e) => e.fmt(f),
            ImageCliError::IoError(e) => e.fmt(f),
        }
    }
}
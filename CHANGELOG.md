
# Unreleased

# 0.2.1 (2021-01-23)

* Bug fix: if the user provides both width and height to `resize` then output an image of size (width, height),
  rather than always preserving the input image's aspect ratio.

# 0.2.0 (2019-12-29)

* New operations:
  - Crop
  - Overlay
  - Map
  - Sequence
  - Pad
  - Tile
  - New
* Improved error reporting.
* Added support for glob inputs and a variable number of output images.
* Moved user guide into a separate file, replaced --generate-readme option
  with --generate-guide, and made minor improvements to the guide and README.

# 0.1.0 (2019-09-15)

Initial release.

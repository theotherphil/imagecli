# Contributing

Please raise any feature requests or bugs as github issues on this repo.

### Tests

This library has a handful of unit tests (which can be run via `cargo test`,
as usual), but does not currently have a regression test suite. When
generating documentation we run all of the examples, so the best current option
for checking for diffs is to run the following command line.

```
cargo run --release -- --generate-readme
```

### Adding new image operations

To add a new image operation, you need to:
* Add a new type implementing `ImageOp` to image_ops.rs. See `Scale` as a simple
  existing example.
* Implement `parse` and `documentation` for your new type. Use the `impl_parse`
  macro for this unless you have good reason not to.
* Add the new op to the list in `image_ops::parse_image_op`.
* Add the new op to `image_ops::documentation`.
* Regenerate README.md using the command line above.

### Updating the README

Do not edit README.md directly - instead update README_template.txt and
run the command line above.

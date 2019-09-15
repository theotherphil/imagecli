# imagecli

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/theotherphil/imagecli/blob/master/LICENSE.md)

A command line image processing tool, built on top of [image] and [imageproc].

$TABLE_OF_CONTENTS

This application applies a series of operations to one or more input images and produces one or
more output images.

Throughout this documentation we will use the following image as input:

<img src='images/robin.png' alt='robin.png'/>

## Basic usage

The simplest possible usage loads an image and then saves an identical copy of it.

$EXAMPLE(
  style: longform
)$

To be slightly more useful, we can specify a different format for the output image. (See [image]
for a list of supported formats.)

$EXAMPLE(
  style: longform
  output_file_extension: jpg
)$

To make the remaining examples less verbose, we'll switch to using the short forms
`-i` and `-o` of `--input` and `--output`.

To do anything more interesting than changing image formats, we need to define an image processing
pipeline via `--pipeline`, or `-p`. This chains together a series of one or more
[operations](#operations).

For example, the following command line converts an image to grayscale.

$EXAMPLE(
  pipeline: gray
)$

And the following rotates an image about its center by 45 degrees.

$EXAMPLE(
  pipeline: rotate 45
)$

## Multi-stage pipelines

You can apply multi operations in a row by chaining them together using `>`. For example, the
following pipeline converts an image to grayscale, rotates it by 30 degrees, and then computes
its gradient using the Sobel filter.

$EXAMPLE(
  pipeline: gray > rotate 30 > sobel
)$

## The image stack

All of the pipelines shown thus far have taken a single image as input and produced a single image
as output. However, we support operations mapping multiple inputs to a single output, as well
as operations mapping a single input to multiple outputs. This is handled via an implicit image
stack: all input images are pushed onto the top of an image stack, and each operation pops one or
more images from the top of the stack, applies some transformation, and pushes one or more output
images back onto the stack. All images provided via the command line are pushed onto the image
stack before we starting running the pipeline, and when the pipeline completes we walk along
the list of output paths in parallel to the images remaining in the stack, saving each image to
the provided path.

### Multiple inputs, single output

The following command line takes two images as input, and applies the `hcat` operator, which
horizontally concatenates a pair of images.

$EXAMPLE(
  num_inputs: 2
  pipeline: hcat
)$

$STACK_DIAGRAM(
  input images are pushed onto the stack
  STACK robin robin_gray
  hcat pops two images
  STACK
  hcat computes result and pushes it onto the stack
  STACK result
  result is saved to the specified output path
)$

### Single input, multiple outputs

There aren't currently any image processing operations that produce multiple outputs from a
single input. However, pipelines can specify `stack operations` that directly manipulate the
image stack. These all have upper case names to make it easier to distinguish between image
processing operations and stack operations.

For example, the `DUP` operator duplicates the top element of the stack. The following example
loads a single image and then saves two copies of it.

$EXAMPLE(
  num_outputs: 2
  pipeline: DUP
)$

$STACK_DIAGRAM(
  input image is pushed onto the stack
  STACK robin
  DUP duplicates the top of the stack
  STACK robin robin
  results are saved to the specified output paths
)$

### Multiple inputs, multiple outputs

As described above, each operation in a pipeline pops a fixed number of images from the top
of the stack, applies a transformation to these images and pushes the results back onto the stack.
This means that the following example only applies the specified blur function to the first
input image.

$EXAMPLE(
  num_inputs: 2
  num_outputs: 2
  pipeline: gaussian 5.0
)$

(Note that in the following diagrams we combine popping from the stack, applying a transformation,
and pushing the result into a single step.)

$STACK_DIAGRAM(
  input images are pushed onto the stack
  STACK robin robin_gray
  gaussian pops the top of the stack, transforms it, and pushes the result
  STACK results robin_gray
  results are saved to the specified output paths
)$

This may not be what you wanted! If you want to apply the `gaussian` operation to both of the two
images in the stack you have two options. The verbose option uses the `SWAP` stack operator to
manually swap the order of the two elements in the stack. `SWAP` is an alias for `ROT 2`, where the
`ROT` operator rotates the positions of the top `n` elements of the stack - the top element moves
`n` positions down the stack and the other the other top elements on the stack move up one.

$STACK_DIAGRAM(
  STACK first second third
  ROT 2
  STACK second first third
  ROT 3
  STACK first third second
  ROT 3
  STACK third second first
)$

The following command line uses `SWAP` to apply a Gaussian blur to both input images.
Notice the second `SWAP` operation, which ensures that the outputs are in the same order as the
inputs.

$EXAMPLE(
  num_inputs: 2
  num_outputs: 2
  pipeline: gaussian 5.0 > SWAP > gaussian 5.0 > SWAP
)$

$STACK_DIAGRAM(
  push input images
  STACK robin robin_gray
  the first gaussian operation transforms the top of the stack
  STACK robin_blurred robin_gray
  SWAP swaps the order of the two stack elements
  STACK robin_gray robin_blurred
  the second gaussian operation transforms the top of the stack
  STACK robin_gray_blurred robin_blurred
  SWAP swaps the order of the two stack elements
  STACK robin_blurred robin_gray_blurred
  save result
)$

As manually rotating through the image stack can be a bit verbose, we also support an array syntax
which applies the nth in a series of operations to the nth element in the stack. For example,
the following command line applies a Gaussian blur to the first image, and a blur with larger
radius to the second.

$EXAMPLE(
  num_inputs: 2
  num_outputs: 2
  pipeline: [gaussian 2.0, gaussian 6.0]
)$

The description above assumes that each operation in the array consumes a single input and produces
a single result. Array operations are actually more general than this, as the operations within
them may consume more than one input or produce more than one result. In this case each operator
is applied to the stack in turn, and the results pushed by each operator are popped into temporary
storage before applying the next operator. Finally, all of the results are pushed to the stack.
`DUP n` is treated as consuming 1 image and creating `n + 1` results, and `ROT n` is always treated
as producing no outputs.

If this explanation isn't clear then work through the stack diagram for the example below. Or
don't - you'll probably never have cause to use this behaviour!

$EXAMPLE(
  num_inputs: 3
  inputs: yellow.png robin.png robin_gray.png
  pipeline: [DUP, hcat] > [vcat, id] > hcat
)$

$STACK_DIAGRAM(
  push input images
  STACK yellow robin robin_gray
  DUP is applied to the first image in the stack, and hcat to the remaining two
  STACK yellow yellow robins
  vcat is applied to the first two images in the stack, and id to the last image
  STACK yellows robins
  the two images are horizontally concatenated
  STACK yellows_robins
  save results
)$

## User-defined functions

We provide limited support for user-defined functions via the `func`, `func2` and `func3`
operators. These operators allow you to specify a function to run on each subpixel of an image.
Functions are arithmetic expression defined in terms of the following components:
* Binary arithmetic operators `+`, `-`, `/`, `*` and `^`.
* Parentheses '(' and ')'.
* Numerical constants, e.g. `4.0`.
* Coordinate variables `x` and `y`. `x` increases from left to right and `y` from top to bottom.
* Variables `p`, `q` and `r`. `p` is the value of the current subpixel in the first image,
  `q` and `r` in the second and third images.
* Comparison operators `<`, `>` and `=`. These evaluate to `1.0` if true and `0.0` if false.

The following function applies a diagonal gradient to an image, increasing its
brightness towards its bottom right.

$EXAMPLE(
  pipeline: func { p + x / 2 + y / 2 }
)$

The following example converts an image to grayscale and then applies a binary threshold. (This is
equivalent to `thresh 120`, but takes longer to run and requires first converting to
grayscale - otherwise the threshold would be applied independently to each channel.)

$EXAMPLE(
  pipeline: gray > func { 255 * (p > 120) }
)$

Our final example uses `func2` to apply a user-defined function to a pair of images.

$EXAMPLE(
  pipeline: DUP 2 > const 184 268 (255, 255, 0) > DUP > ROT 3 > func2 { (p + q) / 2 } > ROT 3 > hcat 3
)$

Current limitations:
* All input images to `func2` and `func3` are required to be the same size.
* Applying different functions to each subpixel (or a function from pixels to pixels) is not
  yet supported.
* All images are converted to RGBA before applying the function.
* There is not yet support for calling out to other functions (e.g. min, max, sin).

## Operations

The following conventions are used to describe the arguments taken by each operation.

* `<foo>`: a required positional argument. For example, `rotate <angle>` accepts the input `rotate 17`.
* `[foo]`: an optional positional argument. For example, `hcat [count]` accepts both `hcat` and `hcat 5`.
* `, ..`: repetition - one or more occurrences of the preceding argument, separated by commas. For example, `op <val>, ..` accepts `op 1` and `op 1, 2`.
* `(LEFT|RIGHT)`: something that matches either LEFT, or RIGHT. For example, `resize (<width> <height>|w=<width>)` accepts both `resize 100 100` and `resize w=100`.
* `'T'`: the literal character `T`, if `T` is given special meaning above. For example, `op '[' <val> ']'` accepts the input `op [ 10 ]`.

Follow the links for a more detailed description, including any restrictions on the inputs (for example that a value
must be an integer, or must be strictly positive).

$OPERATIONS

[image]: https://github.com/image-rs/image
[imageproc]: https://github.com/image-rs/imageproc

imagecli
====

A command line image processing tool, built on top of [image](https://github.com/image-rs/image) and [imageproc](https://github.com/image-rs/imageproc).

Very WIP.

# Examples

All examples use the following input image.

<img src="images/robin.jpg" alt="Input"/>

<pre>
cargo run --release -- -v -i images/robin.jpg -o images/1.png -p 'gray > gaussian 5.0 > scale 0.7 > rotate 45'
</pre>

<img src="images/1.png" alt="Output"/>

<pre>
cargo run --release -- -v -i images/robin.jpg -o images/2.png -p 'sobel'
</pre>

<img src="images/2.png" alt="Output"/>

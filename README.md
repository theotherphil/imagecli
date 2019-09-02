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

<pre>
cargo run --release -- -v -i images/robin.jpg -o images/3.png -p 'carve 0.85'
</pre>

<img src="images/3.png" alt="Output"/>

<pre>
cargo run --release -- -v -i images/robin.jpg -o images/4.png -p 'scale 0.4 > DUP > athresh 10 > SWAP > othresh > hcat'
</pre>

<img src="images/4.png" alt="Output"/>

<pre>
cargo run --release -- -v -i images/robin.jpg -o images/5.png -p 'scale 0.4 > DUP 3 > gaussian 15.0 > ROT 4 > gaussian 10.0 > ROT 3 > gaussian 5.0 > ROT 2 > grid 2 2'
</pre>

<img src="images/5.png" alt="Output"/>

<pre>
cargo run --release -- -v -i images/robin.jpg -o images/6.png -p 'scale 0.5 > median 6 6'
</pre>

<img src="images/6.png" alt="Output"/>

<pre>
cargo run --release -- -v -i images/robin.jpg images/robin_gray.jpg -o images/7.png -p 'hcat'
</pre>

<img src="images/7.png" alt="Output"/>

<pre>
cargo run --release -- -v -i images/robin.jpg images/robin_gray.jpg -o images/8.png images/9.png -p 'scale 0.5 > rotate 10 > SWAP > scale 0.5 > rotate 20 > SWAP'
</pre>

<img src="images/8.png" alt="Output"/>
<img src="images/9.png" alt="Output"/>

<pre>
cargo run --release -- -v -i images/robin.jpg -o images/10.png -p 'scale 0.4 > DUP 3 > [gaussian 1.0, gaussian 3.0, gaussian 5.0, gaussian 7.0] > grid 2 2'
</pre>

<img src="images/10.png" alt="Output"/>

<pre>
cargo run --release -- -v -i images/robin.jpg -o images/11.png -p 'DUP 2 > blue > ROT 3 > green > ROT 2 > red > hcat > hcat'
</pre>

<img src="images/11.png" alt="Output"/>

<pre>
cargo run --release -- -v -i images/robin.jpg -o images/12.png -p 'DUP 3 > [red, green, blue] > hcat > hcat > ROT 2 > hcat'
</pre>

<img src="images/12.png" alt="Output"/>

<pre>
cargo run --release -- -v -i images/robin.jpg -o images/13.png -p 'scale 0.2 > DUP 8 > [scale 1.0, scale 0.9, scale 0.9, scale 0.7, scale 0.6, scale 0.5, scale 0.4, scale 0.3, scale 0.2] > grid 3 3'
</pre>

<img src="images/13.png" alt="Output"/>
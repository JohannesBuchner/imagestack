================================
Image Stack Compression
================================

You have a large set/database of somewhat similar images (same size, similar content) and want to compress them.
This script demonstrates some utilities.

Compression methods
--------------------

* hdf5: Simply stores the data cube as hdf5 array.
* hdf5gz: gzip
* hdf5gzshuffle: gzip+shuffle
* group: loads the hdf5gzshuffle output, groups the images into similar piles using imagehashes, then creates hdf5 data sets for each pile.
* video: stores the frames in a video
  * mp4 (lossy)
  * x264-fast (loss-less)
  * x264-slow (loss-less, smaller size)
  * x264-interm (loss-less)
* groupvideo: groups the images using image hashes, then concatenates the similar images into a video. The video algorithm can take advantage of adjoint similar frames. Same subcommands as video.
* pcaself: PCA of the images, stores the 20 most important eigenvectors and eigenvalues.


Examples
-----------------

	python gen.py hdf5
	python gen.py hdf5gzshuffle
	python gen.py pcaself
	python gen.py checkpca testpcaself.hdf5
	python gen.py checkpca testpcaselfcut.hdf5

	python gen.py checkpca testpcaselfcut.hdf5

	python gen.py hdf5gzshuffle

	python gen.py video x264-fast
	python gen.py video x264-slow
	python gen.py checkvideo test3.mp4

	python gen.py groupvideo x264-fast
	python gen.py groupvideo x264-slow


Results
---------------

2000 images with 300x200 pixels are generated. Grayscale.

.. table:: Results

	+-----------------------+------------+
	| Method                |  Size (KB) |
	+=======================+============+
	| HDF5 gzip+shuffle     |     278728 |
	+-----------------------+------------+
	| Video MP4 (fast)      |      13168 |
	+-----------------------+------------+
	| GroupVideo MP4 (fast) |      12832 |
	+-----------------------+------------+
	| Video MP4 (slow)      |       8308 |
	+-----------------------+------------+
	| GroupVideo MP4 (slow) |       6676 |
	+-----------------------+------------+

Storing a large set of images in a video file can save a lot of space!




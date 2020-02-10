Full HD Raw Video Sequences
Image and Video Compression Lab (LMT@TUM)
Lehrstuhl fuer Medientechnik
Technische Universitaet Muenchen

Notes:
1. The raw video sequence is in YUV420 format, which means the U and V components are downsampled by a factor of 2. For our case, the resolution of Y is 1920*1080, the resolution of U and V is 960*540. The meta information (resolution, frame-rate, data format, bit depth) are in the file name.

2. When we use YUV420 format video sequences for testing, usually we DO NOT convert them to RGB domain and then evaluate the PSNR, usually directly evaluate the PSNR for Y, U and V separately.

3. The RGB2YUV and YUV2RGB conversion are slightly different from the RGB2YCbCr and YCbCr2RGB functions which we implemented in the lab. For easy storage and processing purpose, offsets are adopted and the range of U and V is also 0 to 255. Check here (https://en.wikipedia.org/wiki/YCbCr) ITU-R BT.601 conversion for more details.

4. An example script YUV2Bmp.m is provided to show how to read *.yuv file and how to convert them to RGB images. Check the notes in the script for more details.
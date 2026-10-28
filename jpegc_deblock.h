// jpegc_deblock: a neural JPEG artifact filter for decoded RGBA images, as one C++17 source file
// plus a generated weights header. No dependencies beyond the standard library and the platform's
// thread API; no C++ exceptions are thrown, caught or required (it builds with exceptions
// disabled), no RTTI; every failure is a status code.
//
// The filter is a small MLP (JPEGC_DEBLOCK_NWEIGHTS in jpegc_deblock_weights.h, 2241 for the
// shipped 7x7 net) that reads a window of the image's luma around each pixel, the pixel's position
// inside its 8x8 JPEG block and the file's libjpeg-style quality factor, and predicts a luma
// correction that is added to R, G and B. In JFIF YCbCr that is a correction to Y with Cb and Cr
// unchanged (the Y weights sum to 1, the Cb and Cr weights to 0), up to the per-channel rounding
// and clipping at 0 / 255. Trained by jpegc_train (see the jpegc README).
//
// What the caller must know:
//   - The luma the net reads is recomputed from the RGB (the rounded JFIF Y), since an RGBA buffer
//     no longer carries the decoder's own Y plane. Against jpegc_filter run on the .jpg itself
//     (which reads the decoder's Y) the result differs by at most about 1 LSB in a few pixels.
//   - Pixel (0, 0) must lie on the JPEG's 8x8 block grid: the net reads the block phase from the
//     pixel coordinates. Pass the whole decoded image, not an unaligned sub-rectangle.
//   - Straight (non-premultiplied) alpha only: the correction is added to RGB regardless of alpha.
//   - The net was trained at qualities 30..95; outside that range it extrapolates.
//   - The output is byte-identical to the reference only when the compiler does not contract
//     multiply-adds into FMAs (MSVC /fp:precise, gcc / clang -ffp-contract=off); with FMA
//     contraction or fast-math the output can differ by 1 LSB here and there.
#pragma once
#include <stddef.h>
#include <stdint.h>

enum jpegc_deblock_status {
    JPEGC_DEBLOCK_OK = 0,
    JPEGC_DEBLOCK_BAD_ARGUMENT = 1,   // a null pointer, a bad size, stride, quality, strength; nothing written
    JPEGC_DEBLOCK_OUT_OF_MEMORY = 2   // the luma plane (width x height bytes) could not be allocated; nothing written
};

// Filters a W x H RGBA image (4 bytes per pixel, R first in memory; alpha is copied through) into
// `out`. Strides are in bytes. `out` may be the same buffer as `in` with the same stride (the
// filter is applied in place); any other overlap is not allowed.
//   quality   the libjpeg-style quality factor (1..100) the image was decoded from; use
//             jpegc_deblock_estimate_quality() on the JPEG bytes, or 50 if unknown
//   strength  50 = the filter as trained; 25 halves the correction, 100 doubles it (1..300)
//   threads   worker threads for the pass (0 or less = the hardware thread count; 1 = the
//             calling thread only), created with the platform's thread API and joined inside the
//             call. A thread that cannot be created is not an error: its rows are done on the
//             calling thread, so the call always completes the whole image when it returns OK.
// Returns JPEGC_DEBLOCK_OK, or a status with `out` untouched.
jpegc_deblock_status jpegc_deblock_rgba(const uint8_t* in, int width, int height, int in_stride,
                                        uint8_t* out, int out_stride, int quality, int strength, int threads);

// The libjpeg-style quality factor (1..100) a JPEG file was written with, estimated from its luma
// quantization table (a least-squares fit of the scale factor against the Annex K table, so it is
// exact for libjpeg / stb_image_write / Paint.NET-style encoders and approximate for others). The
// first table with id 0 before the first scan is taken as the luma table. Returns 0 if the bytes
// are not a JPEG, are malformed, or carry no such table; every read is bounds-checked.
int jpegc_deblock_estimate_quality(const uint8_t* jpeg_bytes, size_t size);

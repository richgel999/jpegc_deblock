# jpegc_deblock: the neural JPEG artifact filter as a drop-in library

Three files, no dependencies beyond the standard library and the platform's thread API, no C++
exceptions or RTTI (it builds with both disabled), C++17:

| file | what |
|---|---|
| `jpegc_deblock.h` | the API: two functions and a status enum; every caller caveat is in its comments |
| `jpegc_deblock.cpp` | the filter, the thread pool (`_beginthreadex` / `pthread_create`, status codes) and the JPEG quality estimator |
| `jpegc_deblock_weights.h` | generated: the trained net's shape, gains and 2241 weights as constants (`jpegc_gen_header weights/win7_fd.json lib/jpegc_deblock_weights.h` in the jpegc repo) |

Compile `jpegc_deblock.cpp` as C++17 and link the thread library on POSIX (`-pthread`). For output
byte-identical to the reference, do not let the compiler contract multiply-adds into FMAs
(MSVC `/fp:precise`, gcc / clang `-ffp-contract=off`); otherwise expect occasional 1-LSB differences.

## Usage

```cpp
#include "jpegc_deblock.h"

// jpeg_bytes / jpeg_size: the file; rgba: its decode, W x H, 4 bytes per pixel, R first, stride in bytes
int quality = jpegc_deblock_estimate_quality(jpeg_bytes, jpeg_size);   // 0 = not a JPEG / no table
if (quality == 0) quality = 50;
jpegc_deblock_status st = jpegc_deblock_rgba(rgba, W, H, W * 4,        // input
                                             rgba, W * 4,              // output: in place here, or another buffer
                                             quality, 50, 0);          // strength 50 = as trained; 0 = all hardware threads
if (st != JPEGC_DEBLOCK_OK) { /* JPEGC_DEBLOCK_BAD_ARGUMENT or JPEGC_DEBLOCK_OUT_OF_MEMORY; nothing was written */ }
```

- `strength`: 50 is the trained filter, 25 half the correction, 100 double, up to 300.
- `threads`: created and joined inside the call; a thread that cannot start is not an error, its
  rows are done on the calling thread.
- The image must be the whole decoded JPEG (pixel (0, 0) on the 8x8 block grid), straight alpha.
- The net was trained at qualities 30..95 on photographs; it reads the luma it needs from the RGB.

## What it is

A 7x7-window MLP (52 inputs: the luma window as differences from the centre, the centre's luma,
the 8x8 block phase, the quality factor; hidden 32 and 16, leaky ReLU; one linear output) that
predicts a luma correction added to R, G and B, equivalent to a Y-only correction in YCbCr.
Trained by finite differences on 514 PNGs round-tripped through baseline JPEG at qualities 30..95.
On held-out images it gains about +0.95 dB luma PSNR at quality 30, +0.9 at 40, +0.7 at 80.
About 20 Mpx/s on 32 threads, 1.6 Mpx/s on one.

## Verification

`jpegc_deblock_test` in the jpegc repo proves the library byte for byte against the trainer's
reference on a real JPEG (1 thread and all threads, alpha and row padding untouched), on tiny
synthetic images (1x1, 5x3, 7x100, 3 rows on 64 threads), in place, with every bad argument
refused and the output untouched, and the estimator on empty, truncated and random bytes: 20
checks, passing on Windows (MSVC, Release and Debug) and Linux (gcc, `-fno-exceptions -fno-rtti`),
with identical output bytes across the two. The code was reviewed independently after its first
version; the findings (float literals the generator could emit that do not compile, thread-pool
paths that could terminate the process, exceptions escaping the API, int overflow in the argument
checks) are fixed.

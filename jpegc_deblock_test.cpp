// jpegc_deblock_test: the standalone library against the trainer's reference.
//   jpegc_deblock_test IN.jpg [OUT.png] [--weights W.json]
// Decodes the file with stb_image (RGB), builds an RGBA copy with a non-trivial alpha and stride,
// filters it through jpegc_deblock_rgba() with 1 thread and with all threads, and compares every
// byte against apply_luma() + apply_residual() of src/jc_filter.cpp on the same RGB at the same
// quality (the library's estimate is also compared against jpeg_estimate_quality()). Then the
// corner cases: tiny synthetic images (1x1, 5x3, 7x100, 3 rows on many threads), in == out,
// bad arguments leaving the output untouched, and the estimator on truncated and random bytes.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "jpegc_deblock.h"
#include "jc_filter.h"
#include "jc_image.h"
#include "jc_mlp.h"
#include "jc_noise.h"

static int g_fails = 0;
static void check(bool ok, const char* what) { printf("  %-50s %s\n", what, ok ? "PASS" : "FAIL"); g_fails += !ok; }

// The reference on an RGB image at quality q, strength 50.
static void reference(const FilterDesc& d, const MLP& m, const Image8& rgb, int q, Image8& out) {
    Plane8 y; luma_plane_of(rgb, y);
    std::vector<float> r;
    apply_luma(d, m, y, q, r, 1);
    apply_residual(rgb, r, out);
}
// An RGBA buffer (stride = W*4 + pad) of an RGB image, alpha = x*7 + y*13.
static std::vector<uint8_t> rgba_of(const Image8& rgb, int pad, int& stride) {
    stride = rgb.w * 4 + pad;
    std::vector<uint8_t> v((size_t)stride * rgb.h, 0xEE);
    for (int y = 0; y < rgb.h; y++) for (int x = 0; x < rgb.w; x++) {
        const uint8_t* p = rgb.at(x, y); uint8_t* o = &v[(size_t)y * stride + (size_t)x * 4];
        o[0] = p[0]; o[1] = p[1]; o[2] = p[2]; o[3] = (uint8_t)(x * 7 + y * 13);
    }
    return v;
}
// Pixels of `buf` (stride) that differ from `ref` in RGB, and alpha bytes that are not x*7 + y*13.
static size_t diff_of(const std::vector<uint8_t>& buf, int stride, const Image8& ref, size_t& bad_alpha) {
    size_t bad = 0; bad_alpha = 0;
    for (int y = 0; y < ref.h; y++) for (int x = 0; x < ref.w; x++) {
        const uint8_t* e = ref.at(x, y); const uint8_t* o = &buf[(size_t)y * stride + (size_t)x * 4];
        bad += o[0] != e[0] || o[1] != e[1] || o[2] != e[2];
        bad_alpha += o[3] != (uint8_t)(x * 7 + y * 13);
    }
    return bad;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "jpegc_deblock_test IN.jpg [OUT.png] [--weights W.json]\n"); return 1; }
    std::string weights = "weights/win7_fd.json", out_path = argc > 2 && argv[2][0] != '-' ? argv[2] : "";
    for (int i = 2; i + 1 < argc; i++) if (std::string(argv[i]) == "--weights") weights = argv[i + 1];
    std::string why;
    std::vector<uint8_t> bytes;
    if (!read_file(argv[1], bytes, why)) { fprintf(stderr, "%s\n", why.c_str()); return 1; }
    Image8 rgb;
    if (!jpeg_decode_rgb(bytes.data(), bytes.size(), rgb, why)) { fprintf(stderr, "%s\n", why.c_str()); return 1; }
    int table[64];
    if (!jpeg_luma_qtable(bytes.data(), bytes.size(), table, why)) { fprintf(stderr, "%s\n", why.c_str()); return 1; }
    FilterDesc d; MLP m; std::string trained;
    if (!read_weights_json(weights, d, m, trained, why)) { fprintf(stderr, "%s\n", why.c_str()); return 1; }

    // ---- the image: the estimator, then the filter with 1 thread and with all threads
    const int q_ref = jpeg_estimate_quality(table), q_lib = jpegc_deblock_estimate_quality(bytes.data(), bytes.size());
    printf("quality : reference %d, library %d\n", q_ref, q_lib);
    check(q_ref == q_lib, "estimator equals the reference");
    Image8 ref; reference(d, m, rgb, q_ref, ref);
    const int W = rgb.w, H = rgb.h;
    int in_stride; std::vector<uint8_t> in = rgba_of(rgb, 12, in_stride);
    const int out_stride = W * 4 + 4;
    std::vector<uint8_t> out((size_t)out_stride * H, 0x11);
    for (int threads : { 1, 0 }) {
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(in.data(), W, H, in_stride, out.data(), out_stride, q_ref, 50, threads);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        size_t bad_a = 0; const size_t bad = ok ? diff_of(out, out_stride, ref, bad_a) : (size_t)W * H;
        char what[128]; snprintf(what, sizeof(what), "%dx%d, threads %d: %zu px differ, %.1f ms (%.1f Mpx/s)", W, H, threads, bad, ms, (double)W * H / ms * 1e-3);
        check(ok && bad == 0 && bad_a == 0, what);
    }
    size_t pad_bad = 0;
    for (int y = 0; y < H; y++) for (int k = W * 4; k < out_stride; k++) pad_bad += out[(size_t)y * out_stride + k] != 0x11;
    check(pad_bad == 0, "row padding of the output untouched");
    if (!out_path.empty()) {
        Image8 o; o.resize(W, H);
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) memcpy(o.at(x, y), &out[(size_t)y * out_stride + (size_t)x * 4], 3);
        if (!save_png(out_path, o, why)) { fprintf(stderr, "%s\n", why.c_str()); return 1; }
        printf("output  : %s\n", out_path.c_str());
    }

    // ---- tiny synthetic images (hash noise), several strides, 1 thread and 64 threads
    {
        const int sizes[][2] = { { 1, 1 }, { 5, 3 }, { 7, 100 }, { 100, 3 }, { 9, 9 }, { 17, 40 } };
        for (const auto& s : sizes) {
            Image8 img; img.resize(s[0], s[1]);
            for (size_t i = 0; i < img.rgb.size(); i++) img.rgb[i] = (uint8_t)(jc_noise_u32(7, 1, (uint32_t)s[0], (uint32_t)s[1], i) >> 24);
            Image8 r; reference(d, m, img, 42, r);
            for (int threads : { 1, 64 }) {
                int st; std::vector<uint8_t> a = rgba_of(img, 8, st);
                std::vector<uint8_t> o((size_t)st * img.h, 0x33);
                const bool ok = JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(a.data(), img.w, img.h, st, o.data(), st, 42, 50, threads);
                size_t bad_a = 0; const size_t bad = ok ? diff_of(o, st, r, bad_a) : 1;
                char what[96]; snprintf(what, sizeof(what), "%dx%d on %d threads", img.w, img.h, threads);
                check(ok && bad == 0 && bad_a == 0, what);
            }
        }
    }
    // ---- in == out (the same stride): the filter in place
    {
        Image8 crop; crop.resize(std::min(W, 64), std::min(H, 48));
        for (int y = 0; y < crop.h; y++) memcpy(crop.at(0, y), rgb.at(0, y), (size_t)crop.w * 3);
        Image8 r; reference(d, m, crop, q_ref, r);
        int st; std::vector<uint8_t> a = rgba_of(crop, 4, st);
        const bool ok = JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(a.data(), crop.w, crop.h, st, a.data(), st, q_ref, 50, 0);
        size_t bad_a = 0; const size_t bad = ok ? diff_of(a, st, r, bad_a) : 1;
        check(ok && bad == 0 && bad_a == 0, "in == out, same stride (in place)");
    }
    // ---- bad arguments: refused, the output untouched
    {
        std::vector<uint8_t> o((size_t)out_stride * H, 0x55), o0 = o;
        bool any = false;
        any |= JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(nullptr, W, H, in_stride, o.data(), out_stride, 50, 50, 1);
        any |= JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(in.data(), W, H, in_stride, nullptr, out_stride, 50, 50, 1);
        any |= JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(in.data(), 0, H, in_stride, o.data(), out_stride, 50, 50, 1);
        any |= JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(in.data(), W, -1, in_stride, o.data(), out_stride, 50, 50, 1);
        any |= JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(in.data(), W, H, W * 4 - 1, o.data(), out_stride, 50, 50, 1);
        any |= JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(in.data(), W, H, in_stride, o.data(), W * 4 - 4, 50, 50, 1);
        any |= JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(in.data(), W, H, in_stride, o.data(), out_stride, 0, 50, 1);
        any |= JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(in.data(), W, H, in_stride, o.data(), out_stride, 101, 50, 1);
        any |= JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(in.data(), W, H, in_stride, o.data(), out_stride, 50, 0, 1);
        any |= JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(in.data(), W, H, in_stride, o.data(), out_stride, 50, 301, 1);
        any |= JPEGC_DEBLOCK_OK == jpegc_deblock_rgba(in.data(), 0x7FFFFFFF, 1, 0x7FFFFFFF, o.data(), 0x7FFFFFFF, 50, 50, 1);   // width * 4 overflows int
        check(!any && o == o0, "bad arguments refused, output untouched");
    }
    // ---- the estimator on hostile bytes: never reads past the buffer, returns 0 or a sane value
    {
        bool ok = jpegc_deblock_estimate_quality(nullptr, 0) == 0 && jpegc_deblock_estimate_quality(bytes.data(), 0) == 0 && jpegc_deblock_estimate_quality(bytes.data(), 2) == 0;
        for (size_t n = 1; n < std::min<size_t>(bytes.size(), 4096); n++) {   // every truncation of the real file's head (a copy: nothing readable past n)
            std::vector<uint8_t> t(bytes.begin(), bytes.begin() + (ptrdiff_t)n);
            const int q = jpegc_deblock_estimate_quality(t.data(), t.size());
            ok = ok && q >= 0 && q <= 100;
        }
        for (int trial = 0; trial < 2000; trial++) {   // random bytes, some with a JPEG start and a DQT marker planted
            std::vector<uint8_t> t(4 + (jc_noise_u32(9, 2, (uint32_t)trial, 0, 0) % 300));
            for (size_t i = 0; i < t.size(); i++) t[i] = (uint8_t)(jc_noise_u32(9, 3, (uint32_t)trial, 0, i) >> 24);
            if (trial & 1) { t[0] = 0xFF; t[1] = 0xD8; t[2] = 0xFF; t[3] = 0xDB; }
            const int q = jpegc_deblock_estimate_quality(t.data(), t.size());
            ok = ok && q >= 0 && q <= 100;
        }
        check(ok, "estimator on empty, truncated and random bytes");
        // a DQT whose declared length runs past the end
        std::vector<uint8_t> t = { 0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x43, 0x00 };
        check(jpegc_deblock_estimate_quality(t.data(), t.size()) == 0, "estimator on a DQT that overruns the file");
    }
    printf(g_fails ? "test: %d failed\n" : "test: all passed\n", g_fails);
    return g_fails ? 1 : 0;
}

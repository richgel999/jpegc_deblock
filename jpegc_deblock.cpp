// jpegc_deblock.cpp: see jpegc_deblock.h. The arithmetic mirrors the trainer's reference
// (src/jc_filter.cpp features() / apply_luma() / apply_residual(), src/jc_mlp.h
// mlp_forward_linear(), src/jc_image.cpp luma_plane_of() and the DQT walk) expression for
// expression, so the output is byte-identical to jpegc_filter's on the same RGB input.
#include "jpegc_deblock.h"
#include "jpegc_deblock_weights.h"
#include <algorithm>
#include <cmath>
#include <new>

namespace {

const int kR = JPEGC_DEBLOCK_WIN;
const int kNin = JPEGC_DEBLOCK_NIN;
const float kInv255 = 1.0f / 255.0f;
static_assert(JPEGC_DEBLOCK_NIN == (2 * JPEGC_DEBLOCK_WIN + 1) * (2 * JPEGC_DEBLOCK_WIN + 1) - 1 + (JPEGC_DEBLOCK_IN_ABS ? 1 : 0)
                                   + (JPEGC_DEBLOCK_IN_PHASE == 2 ? 16 : JPEGC_DEBLOCK_IN_PHASE ? 2 : 0) + (JPEGC_DEBLOCK_IN_Q ? 1 : 0),
              "the weights header's input count does not match its feature switches");
static_assert(JPEGC_DEBLOCK_MAXW >= 1 && JPEGC_DEBLOCK_MAXW <= 256, "the activation buffers hold hidden widths of 1..256");

inline float u8f(uint8_t v) { return (float)v * kInv255; }
inline float activate(float x) { return x > 0 ? x : JPEGC_DEBLOCK_LEAK * x; }

// = features() of jc_filter.cpp: the window as differences from the centre (the centre's own
// slot dropped), the centre's absolute luma, the 8x8 block phase, the quality input.
void features(const uint8_t* y, int W, int H, int px, int py, float qin, float* f) {
    const float c0 = u8f(y[(size_t)py * W + px]);
    int k = 0;
    for (int dy = -kR; dy <= kR; dy++) {
        const int yy = std::min(H - 1, std::max(0, py + dy));
        for (int dx = -kR; dx <= kR; dx++) {
            if (dx == 0 && dy == 0) continue;
            const int xx = std::min(W - 1, std::max(0, px + dx));
            f[k++] = u8f(y[(size_t)yy * W + xx]) - c0;
        }
    }
#if JPEGC_DEBLOCK_IN_ABS
    f[k++] = c0 * JPEGC_DEBLOCK_ABS_GAIN;
#endif
#if JPEGC_DEBLOCK_IN_PHASE == 2
    for (int i = 0; i < 8; i++) f[k++] = (px & 7) == i ? JPEGC_DEBLOCK_PHASE_GAIN : 0.0f;
    for (int i = 0; i < 8; i++) f[k++] = (py & 7) == i ? JPEGC_DEBLOCK_PHASE_GAIN : 0.0f;
#elif JPEGC_DEBLOCK_IN_PHASE
    f[k++] = ((float)(px & 7) - 3.5f) * (1.0f / 3.5f) * JPEGC_DEBLOCK_PHASE_GAIN;
    f[k++] = ((float)(py & 7) - 3.5f) * (1.0f / 3.5f) * JPEGC_DEBLOCK_PHASE_GAIN;
#endif
#if JPEGC_DEBLOCK_IN_Q
    f[k++] = qin;
#else
    (void)qin;
#endif
}

// = mlp_forward_linear() of jc_mlp.h with one output.
float forward(const float* in) {
    float bufA[JPEGC_DEBLOCK_MAXW], bufB[JPEGC_DEBLOCK_MAXW];
    const float* p = jpegc_deblock_weights;
    const float* cur = in;
    int ncur = kNin;
    float* nxt = bufA;
    for (int l = 0; l < JPEGC_DEBLOCK_NLAYERS; l++) {
        const int nh = jpegc_deblock_hidden[l];
        const float* W = p; const float* b = W + (size_t)nh * ncur;
        for (int j = 0; j < nh; j++) {
            float s = b[j]; const float* w = W + (size_t)j * ncur;
            for (int i = 0; i < ncur; i++) s += w[i] * cur[i];
            nxt[j] = activate(s);
        }
        p = b + nh;
        cur = nxt; ncur = nh;
        nxt = (nxt == bufA) ? bufB : bufA;
    }
    float s = p[ncur];   // the output bias follows the output row
    for (int i = 0; i < ncur; i++) s += p[i] * cur[i];
    return s;
}

inline uint8_t add_clamp(uint8_t v, float r) {
    const long q = std::lround((float)v + r * 255.0f);
    return (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
}

// ---------------------------------------------------------------- threads
// The platform's thread API, so that a failed start is a return code rather than an exception.
struct ThreadStart { void (*fn)(void*); void* arg; };   // what a new thread runs: fn(arg)
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
struct NativeThread {
    HANDLE h = nullptr;
    static unsigned __stdcall entry(void* p) { const ThreadStart* s = (const ThreadStart*)p; s->fn(s->arg); return 0; }
    bool start(const ThreadStart* s) { h = (HANDLE)_beginthreadex(nullptr, 0, entry, (void*)s, 0, nullptr); return h != nullptr; }   // 0 = could not start
    void join() { if (h) { WaitForSingleObject(h, INFINITE); CloseHandle(h); h = nullptr; } }
};
static int hardware_threads() { SYSTEM_INFO si; GetSystemInfo(&si); return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1; }
#else
#include <pthread.h>
#include <unistd.h>
struct NativeThread {
    pthread_t t; bool started = false;
    static void* entry(void* p) { const ThreadStart* s = (const ThreadStart*)p; s->fn(s->arg); return nullptr; }
    bool start(const ThreadStart* s) { started = pthread_create(&t, nullptr, entry, (void*)s) == 0; return started; }   // an error code = could not start
    void join() { if (started) { pthread_join(t, nullptr); started = false; } }
};
static int hardware_threads() { const long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 1; }
#endif

// Runs fn(ctx, row0, row1) over [0, H) on up to `threads` threads, the calling thread one of
// them. Always completes every row: a thread that cannot be created costs nothing but its rows,
// which the calling thread does after its own; every thread that was started is joined before
// returning. Nothing here can fail in a way that is not handled, so it returns nothing. The
// per-thread bookkeeping (at most kMaxThreads entries) lives on the stack: no allocation.
static const int kMaxThreads = 256;
typedef void (*RowFn)(void* ctx, int r0, int r1);
struct RowJob { RowFn fn; void* ctx; int r0, r1; ThreadStart start; };
static void row_job_entry(void* p) { RowJob* j = (RowJob*)p; j->fn(j->ctx, j->r0, j->r1); }
static void parallel_rows(int H, int threads, RowFn fn, void* ctx) {
    threads = std::min(threads, std::max(1, H / 16));   // at least 16 rows per thread: thread creation is not free
    threads = std::min(threads, kMaxThreads);
    if (threads <= 1) { fn(ctx, 0, H); return; }
    RowJob jobs[kMaxThreads];
    NativeThread pool[kMaxThreads];
    bool started[kMaxThreads];
    for (int t = 1; t < threads; t++) {
        RowJob& j = jobs[t];
        j.fn = fn; j.ctx = ctx; j.r0 = (int)((int64_t)H * t / threads); j.r1 = (int)((int64_t)H * (t + 1) / threads);
        j.start.fn = row_job_entry; j.start.arg = &j;
        started[t] = pool[t].start(&j.start);
    }
    fn(ctx, 0, (int)((int64_t)H / threads));
    for (int t = 1; t < threads; t++) if (!started[t]) fn(ctx, jobs[t].r0, jobs[t].r1);   // the rows of the threads that could not start
    for (int t = 1; t < threads; t++) if (started[t]) pool[t].join();
}

// The two passes' shared state.
struct Ctx {
    const uint8_t* in; uint8_t* out; uint8_t* y;
    int W, H, in_stride, out_stride;
    float qin, rscale;
};
// 1. the luma plane: rounded JFIF Y of the RGB (= luma_plane_of())
static void luma_rows(void* p, int r0, int r1) {
    const Ctx& c = *(const Ctx*)p;
    for (int py = r0; py < r1; py++) {
        const uint8_t* src = c.in + (size_t)py * c.in_stride;
        uint8_t* dst = c.y + (size_t)py * c.W;
        for (int px = 0; px < c.W; px++, src += 4) {
            const long v = std::lround(0.299f * src[0] + 0.587f * src[1] + 0.114f * src[2]);
            dst[px] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    }
}
// 2. the filter: the residual of every pixel, added to R, G, B (= apply_luma() + apply_residual()).
// Each pixel reads its own input bytes before writing its output bytes, so in == out is safe.
static void filter_rows(void* p, int r0, int r1) {
    const Ctx& c = *(const Ctx*)p;
    float f[JPEGC_DEBLOCK_NIN];
    for (int py = r0; py < r1; py++) {
        const uint8_t* src = c.in + (size_t)py * c.in_stride;
        uint8_t* dst = c.out + (size_t)py * c.out_stride;
        for (int px = 0; px < c.W; px++, src += 4, dst += 4) {
            features(c.y, c.W, c.H, px, py, c.qin, f);
            float r = forward(f);
            if (c.rscale != 1.0f) r *= c.rscale;
            const uint8_t r8 = src[0], g8 = src[1], b8 = src[2], a8 = src[3];
            dst[0] = add_clamp(r8, r);
            dst[1] = add_clamp(g8, r);
            dst[2] = add_clamp(b8, r);
            dst[3] = a8;
        }
    }
}

} // namespace

jpegc_deblock_status jpegc_deblock_rgba(const uint8_t* in, int width, int height, int in_stride,
                                        uint8_t* out, int out_stride, int quality, int strength, int threads) {
    if (!in || !out || width < 1 || height < 1) return JPEGC_DEBLOCK_BAD_ARGUMENT;
    if ((int64_t)width * 4 > in_stride || (int64_t)width * 4 > out_stride) return JPEGC_DEBLOCK_BAD_ARGUMENT;
    if (quality < 1 || quality > 100 || strength < 1 || strength > 300) return JPEGC_DEBLOCK_BAD_ARGUMENT;
    if (threads <= 0) threads = hardware_threads();
    Ctx c;
    c.in = in; c.out = out; c.W = width; c.H = height; c.in_stride = in_stride; c.out_stride = out_stride;
    c.qin = ((float)quality - 62.5f) * (1.0f / 32.5f) * JPEGC_DEBLOCK_Q_GAIN;   // = q_input_of()
    c.rscale = (float)strength / 50.0f;
    c.y = new (std::nothrow) uint8_t[(size_t)width * (size_t)height];
    if (!c.y) return JPEGC_DEBLOCK_OUT_OF_MEMORY;   // nothing written
    parallel_rows(height, threads, luma_rows, &c);
    parallel_rows(height, threads, filter_rows, &c);
    delete[] c.y;
    return JPEGC_DEBLOCK_OK;
}

// ---------------------------------------------------------------- the quality estimate
namespace {
const int kZigZag[64] = {
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };
// The Annex K luminance table (natural order), as libjpeg's jcparam.c.
const int kStdLuma[64] = {
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99 };

// = jpeg_luma_qtable(): the luma table (id 0) in the file's zigzag order, from the marker
// segments before the scan. Every read is bounds-checked against n.
bool luma_qtable(const uint8_t* data, size_t n, int table[64]) {
    if (!(n >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF)) return false;
    size_t i = 2;
    while (i + 4 <= n) {
        if (data[i] != 0xFF) return false;
        while (i < n && data[i] == 0xFF) i++;
        if (i >= n) break;
        const int marker = data[i++];
        if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) continue;
        if (marker == 0xD9 || marker == 0xDA) break;
        if (i + 2 > n) break;
        const size_t len = ((size_t)data[i] << 8) | data[i + 1];
        if (len < 2 || len > n - i) return false;
        if (marker == 0xDB) {
            size_t p = i + 2;
            const size_t end = i + len;
            while (p < end) {
                const int pq = data[p] >> 4, tq = data[p] & 15;
                p++;
                const size_t need = pq ? 128 : 64;
                if (need > end - p) return false;
                if (tq == 0) {
                    for (int k = 0; k < 64; k++) table[k] = pq ? (((int)data[p + 2 * k] << 8) | data[p + 2 * k + 1]) : data[p + k];
                    return true;
                }
                p += need;
            }
        }
        i += len;
    }
    return false;
}
} // namespace

int jpegc_deblock_estimate_quality(const uint8_t* jpeg_bytes, size_t size) {
    int table[64];
    if (!jpeg_bytes || !luma_qtable(jpeg_bytes, size, table)) return 0;
    // = jpeg_estimate_quality(): the least-squares scale over the unclamped entries, inverted
    double num = 0, den = 0;
    int used = 0;
    for (int k = 0; k < 64; k++) {
        const int t = table[k], s = kStdLuma[kZigZag[k]];
        if (t <= 1 || t >= 255) continue;
        num += (double)t * s;
        den += (double)s * s;
        used++;
    }
    if (used == 0) {
        int ones = 0;
        for (int i = 0; i < 64; i++) ones += table[i] <= 1;
        return ones > 32 ? 100 : 1;
    }
    const double scale = 100.0 * num / den;
    const double q = scale <= 100.0 ? (200.0 - scale) / 2.0 : 5000.0 / scale;
    const int qi = (int)std::lround(q);
    return qi < 1 ? 1 : qi > 100 ? 100 : qi;
}

#pragma once

// ── Standard headers ──────────────────────────────────────────────────────
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <string>
#include <vector>
#include <tuple>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <algorithm>
#include <sys/stat.h>
#include <cuda_runtime.h>

using namespace std;

// ── stb_image (single-header image loader / writer) ───────────────────────
// Define the implementation exactly once (here, in the header that the .cu
// file includes first) so nvcc compiles the stb function bodies into the
// translation unit without needing a separate .c file.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ── Constants ─────────────────────────────────────────────────────────────
#define HISTOGRAM_SIZE 256       // 8-bit pixel range
#define BLOCK_DIM 16             // 16×16 = 256 threads per block (2-D kernels)
#define BLOCK_SIZE_1D 256        // threads per block for 1-D kernels
#define TILE_DIM (BLOCK_DIM + 4) // 16 + 2 halo pixels on each side (5-tap kernel)

// ── Per-image statistics written to output CSV ────────────────────────────
struct ImageStats
{
    string filename;
    int width;
    int height;
    float blurTimeMs;
    float sobelTimeMs;
    float histEqTimeMs;
    float totalGpuTimeMs;
    float meanPixelIn;  // mean pixel value of input image
    float meanPixelOut; // mean pixel value after histogram equalization
};

// ── CUDA error-checking helper ─────────────────────────────────────────────
inline void cudaCheck(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d — %s\n",
                file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

// ── Kernel declarations ────────────────────────────────────────────────────

// 5×5 Gaussian blur using __constant__ memory for the filter coefficients
__global__ void gaussianBlur(const unsigned char *__restrict__ src,
                             unsigned char *__restrict__ dst,
                             int width, int height);

// 3×3 Sobel edge detection using shared-memory tile loading
__global__ void sobelEdgeDetect(const unsigned char *__restrict__ src,
                                unsigned char *__restrict__ dst,
                                int width, int height);

// Pass 1 of histogram equalisation — build a global 256-bin histogram using
// per-block shared-memory atomics merged into a global array
__global__ void buildHistogram(const unsigned char *__restrict__ src,
                               unsigned int *histogram,
                               int numPixels);

// Pass 2 of histogram equalisation — apply the precomputed LUT to each pixel
__global__ void applyLUT(const unsigned char *__restrict__ src,
                         unsigned char *__restrict__ dst,
                         const unsigned char *__restrict__ lut,
                         int numPixels);

// ── Host function declarations ─────────────────────────────────────────────
__host__ vector<string> listImageFiles(const string &directory);

__host__ ImageStats processImage(const string &inputPath,
                                 const string &outputPath);

__host__ void writeCsvReport(const string &csvPath,
                             const vector<ImageStats> &stats);

__host__ void printSummary(const vector<ImageStats> &stats);
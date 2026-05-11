#include "image_processing.h"

// ════════════════════════════════════════════════════════════════════════════
// CONSTANT MEMORY — 5×5 Gaussian kernel coefficients (normalised to sum = 1)
// Stored in __constant__ memory so all threads in a warp read the same
// value in a single broadcast, with no global memory traffic.
//
//  Unnormalised Pascal / Gaussian 5×5:
//   1  4  6  4  1
//   4 16 24 16  4
//   6 24 36 24  6
//   4 16 24 16  4
//   1  4  6  4  1   sum = 256
// ════════════════════════════════════════════════════════════════════════════
__constant__ float c_gaussKernel[25] = {
     1/256.f,  4/256.f,  6/256.f,  4/256.f,  1/256.f,
     4/256.f, 16/256.f, 24/256.f, 16/256.f,  4/256.f,
     6/256.f, 24/256.f, 36/256.f, 24/256.f,  6/256.f,
     4/256.f, 16/256.f, 24/256.f, 16/256.f,  4/256.f,
     1/256.f,  4/256.f,  6/256.f,  4/256.f,  1/256.f
};

// ════════════════════════════════════════════════════════════════════════════
// KERNEL 1 — Gaussian Blur
//
// Each thread processes one output pixel.  It reads a 5×5 neighbourhood from
// global memory (clamped at borders) and convolves with c_gaussKernel stored
// in constant memory.
//
// Thread / block layout: 2-D, BLOCK_DIM × BLOCK_DIM
// Grid: ceil(width/BLOCK_DIM) × ceil(height/BLOCK_DIM)
// ════════════════════════════════════════════════════════════════════════════
__global__ void gaussianBlur(const unsigned char* __restrict__ src,
                               unsigned char* __restrict__ dst,
                               int width, int height)
{
    // Global pixel coordinates for this thread
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    float sum = 0.0f;
    int kidx = 0;

    // Convolve with the 5×5 kernel; clamp coordinates at image borders
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int sx = min(max(px + kx, 0), width  - 1);
            int sy = min(max(py + ky, 0), height - 1);
            sum += c_gaussKernel[kidx++] * (float)src[sy * width + sx];
        }
    }

    dst[py * width + px] = (unsigned char)min(max((int)sum, 0), 255);
}

// ════════════════════════════════════════════════════════════════════════════
// KERNEL 2 — Sobel Edge Detection
//
// Each thread processes one output pixel using a 3×3 Sobel operator.
// A (BLOCK_DIM+2) × (BLOCK_DIM+2) shared-memory tile is loaded cooperatively
// so the 8-pixel halo only requires one global read per border thread
// rather than redundant reads from every thread that needs a neighbour.
//
// Gx kernel:  Gy kernel:
//  -1  0  1   -1 -2 -1
//  -2  0  2    0  0  0
//  -1  0  1    1  2  1
// ════════════════════════════════════════════════════════════════════════════
__global__ void sobelEdgeDetect(const unsigned char* __restrict__ src,
                                 unsigned char* __restrict__ dst,
                                 int width, int height)
{
    // Shared tile: includes 1-pixel halo on all four sides
    __shared__ unsigned char tile[BLOCK_DIM + 2][BLOCK_DIM + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global pixel this thread is responsible for
    int px = blockIdx.x * BLOCK_DIM + tx;
    int py = blockIdx.y * BLOCK_DIM + ty;

    // Load the tile including halo.  Clamp at borders.
    // Offset by 1 because the tile has a 1-pixel halo.
    int sx = min(max(px, 0), width  - 1);
    int sy = min(max(py, 0), height - 1);
    tile[ty + 1][tx + 1] = src[sy * width + sx];

    // Left halo column
    if (tx == 0) {
        int lx = min(max(px - 1, 0), width - 1);
        tile[ty + 1][0] = src[sy * width + lx];
    }
    // Right halo column
    if (tx == BLOCK_DIM - 1) {
        int rx = min(max(px + 1, 0), width - 1);
        tile[ty + 1][BLOCK_DIM + 1] = src[sy * width + rx];
    }
    // Top halo row
    if (ty == 0) {
        int ty2 = min(max(py - 1, 0), height - 1);
        tile[0][tx + 1] = src[ty2 * width + sx];
    }
    // Bottom halo row
    if (ty == BLOCK_DIM - 1) {
        int by = min(max(py + 1, 0), height - 1);
        tile[BLOCK_DIM + 1][tx + 1] = src[by * width + sx];
    }
    // Four corner halo pixels
    if (tx == 0 && ty == 0) {
        tile[0][0] = src[min(max(py-1,0),height-1)*width + min(max(px-1,0),width-1)];
    }
    if (tx == BLOCK_DIM-1 && ty == 0) {
        tile[0][BLOCK_DIM+1] = src[min(max(py-1,0),height-1)*width + min(max(px+1,0),width-1)];
    }
    if (tx == 0 && ty == BLOCK_DIM-1) {
        tile[BLOCK_DIM+1][0] = src[min(max(py+1,0),height-1)*width + min(max(px-1,0),width-1)];
    }
    if (tx == BLOCK_DIM-1 && ty == BLOCK_DIM-1) {
        tile[BLOCK_DIM+1][BLOCK_DIM+1] = src[min(max(py+1,0),height-1)*width + min(max(px+1,0),width-1)];
    }

    __syncthreads();  // all tile data must be loaded before any thread reads it

    if (px >= width || py >= height) return;

    // Apply Sobel operator using shared memory (no extra global reads)
    int gx = -(int)tile[ty][tx]     + (int)tile[ty][tx+2]
             -2*(int)tile[ty+1][tx] + 2*(int)tile[ty+1][tx+2]
             -(int)tile[ty+2][tx]   + (int)tile[ty+2][tx+2];

    int gy = -(int)tile[ty][tx]   - 2*(int)tile[ty][tx+1]   - (int)tile[ty][tx+2]
             +(int)tile[ty+2][tx] + 2*(int)tile[ty+2][tx+1] + (int)tile[ty+2][tx+2];

    int magnitude = (int)sqrtf((float)(gx*gx + gy*gy));
    dst[py * width + px] = (unsigned char)min(magnitude, 255);
}

// ════════════════════════════════════════════════════════════════════════════
// KERNEL 3a — Build Histogram (Pass 1 of histogram equalisation)
//
// Each block builds a partial 256-bin histogram in shared memory using
// atomicAdd, then merges its partial histogram into the global array.
// Using shared-memory atomics first avoids serialising all threads on the
// same 256 global counters simultaneously.
// ════════════════════════════════════════════════════════════════════════════
__global__ void buildHistogram(const unsigned char* __restrict__ src,
                                unsigned int* histogram,
                                int numPixels)
{
    // Per-block partial histogram in shared memory
    __shared__ unsigned int partialHist[HISTOGRAM_SIZE];

    // Initialise shared histogram to zero
    if (threadIdx.x < HISTOGRAM_SIZE) {
        partialHist[threadIdx.x] = 0u;
    }
    __syncthreads();

    // Each thread processes multiple pixels (strided loop over the image)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < numPixels) {
        atomicAdd(&partialHist[src[i]], 1u);
        i += stride;
    }
    __syncthreads();

    // Merge partial histogram into global histogram
    if (threadIdx.x < HISTOGRAM_SIZE) {
        atomicAdd(&histogram[threadIdx.x], partialHist[threadIdx.x]);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// KERNEL 3b — Apply LUT (Pass 2 of histogram equalisation)
//
// Each thread maps one pixel through the precomputed equalisation LUT.
// The LUT is small (256 bytes) and is loaded into L1/texture cache naturally
// due to its access pattern.
// ════════════════════════════════════════════════════════════════════════════
__global__ void applyLUT(const unsigned char* __restrict__ src,
                          unsigned char* __restrict__ dst,
                          const unsigned char* __restrict__ lut,
                          int numPixels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPixels) {
        dst[i] = lut[src[i]];
    }
}

// ════════════════════════════════════════════════════════════════════════════
// HOST — List all image files in a directory
// ════════════════════════════════════════════════════════════════════════════
__host__ vector<string> listImageFiles(const string& directory)
{
    vector<string> files;
    DIR* dir = opendir(directory.c_str());
    if (!dir) {
        fprintf(stderr, "Cannot open directory: %s\n", directory.c_str());
        return files;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        string name(entry->d_name);
        // Accept .png, .jpg, .jpeg, .pgm, .bmp
        string ext = name.size() > 4 ? name.substr(name.rfind('.')) : "";
        for (auto& c : ext) c = (char)tolower(c);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
            ext == ".pgm" || ext == ".bmp") {
            files.push_back(directory + "/" + name);
        }
    }
    closedir(dir);

    // Sort for deterministic processing order
    sort(files.begin(), files.end());
    return files;
}

// ════════════════════════════════════════════════════════════════════════════
// HOST — Process a single image through the full GPU pipeline
// ════════════════════════════════════════════════════════════════════════════
__host__ ImageStats processImage(const string& inputPath,
                                  const string& outputPath)
{
    ImageStats stats;
    stats.filename = inputPath;

    // ── Load image (CPU) ──────────────────────────────────────────────────
    int width, height, channels;
    // Force grayscale (1 channel) regardless of source format
    unsigned char* h_input = stbi_load(inputPath.c_str(),
                                        &width, &height, &channels, 1);
    if (!h_input) {
        fprintf(stderr, "Failed to load image: %s\n", inputPath.c_str());
        stats.width = stats.height = 0;
        return stats;
    }

    stats.width  = width;
    stats.height = height;
    int numPixels = width * height;
    size_t imgBytes = numPixels * sizeof(unsigned char);

    // Compute mean input pixel value for the CSV report
    double sumIn = 0.0;
    for (int i = 0; i < numPixels; i++) sumIn += h_input[i];
    stats.meanPixelIn = (float)(sumIn / numPixels);

    // ── Allocate device memory ────────────────────────────────────────────
    unsigned char *d_src = NULL, *d_blurred = NULL,
                  *d_edges = NULL, *d_heq = NULL, *d_lut = NULL;
    unsigned int  *d_histogram = NULL;

    CUDA_CHECK(cudaMalloc(&d_src,       imgBytes));
    CUDA_CHECK(cudaMalloc(&d_blurred,   imgBytes));
    CUDA_CHECK(cudaMalloc(&d_edges,     imgBytes));
    CUDA_CHECK(cudaMalloc(&d_heq,       imgBytes));
    CUDA_CHECK(cudaMalloc(&d_lut,       HISTOGRAM_SIZE * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_histogram, HISTOGRAM_SIZE * sizeof(unsigned int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_src, h_input, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_histogram, 0, HISTOGRAM_SIZE * sizeof(unsigned int)));

    // ── Grid/block configuration ──────────────────────────────────────────
    dim3 block2D(BLOCK_DIM, BLOCK_DIM);
    dim3 grid2D((width  + BLOCK_DIM - 1) / BLOCK_DIM,
                (height + BLOCK_DIM - 1) / BLOCK_DIM);

    dim3 block1D(BLOCK_SIZE_1D);
    dim3 grid1D((numPixels + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);

    // ── CUDA timing events ────────────────────────────────────────────────
    cudaEvent_t t0, t1, t2, t3, t4;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&t2));
    CUDA_CHECK(cudaEventCreate(&t3));
    CUDA_CHECK(cudaEventCreate(&t4));

    // ── KERNEL 1: Gaussian Blur ───────────────────────────────────────────
    CUDA_CHECK(cudaEventRecord(t0));
    gaussianBlur<<<grid2D, block2D>>>(d_src, d_blurred, width, height);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaGetLastError());

    // ── KERNEL 2: Sobel Edge Detection ───────────────────────────────────
    CUDA_CHECK(cudaEventRecord(t2));
    sobelEdgeDetect<<<grid2D, block2D>>>(d_blurred, d_edges, width, height);
    CUDA_CHECK(cudaEventRecord(t3));
    CUDA_CHECK(cudaGetLastError());

    // ── KERNEL 3: Histogram Equalisation ─────────────────────────────────
    // Pass 1: build histogram from blurred image
    CUDA_CHECK(cudaEventRecord(t4));
    int histBlocks = min(grid1D.x, 128);  // cap to avoid too many global atomics
    buildHistogram<<<histBlocks, BLOCK_SIZE_1D>>>(d_blurred, d_histogram, numPixels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pass 1 (CPU): compute CDF and build LUT from the histogram
    unsigned int h_hist[HISTOGRAM_SIZE] = {0};
    CUDA_CHECK(cudaMemcpy(h_hist, d_histogram,
                           HISTOGRAM_SIZE * sizeof(unsigned int),
                           cudaMemcpyDeviceToHost));

    // Compute CDF
    unsigned long long cdf[HISTOGRAM_SIZE] = {0};
    cdf[0] = h_hist[0];
    for (int i = 1; i < HISTOGRAM_SIZE; i++) {
        cdf[i] = cdf[i-1] + h_hist[i];
    }

    // Find cdf_min (first non-zero CDF value)
    unsigned long long cdf_min = 0;
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        if (cdf[i] > 0) { cdf_min = cdf[i]; break; }
    }

    // Build equalisation LUT:  lut[v] = round((cdf[v]-cdf_min)/(N-cdf_min) * 255)
    unsigned char h_lut[HISTOGRAM_SIZE] = {0};
    double scale = (numPixels - (double)cdf_min);
    if (scale < 1.0) scale = 1.0;
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        double val = ((double)cdf[i] - (double)cdf_min) / scale * 255.0;
        h_lut[i] = (unsigned char)min(max((int)round(val), 0), 255);
    }

    // Copy LUT to device
    CUDA_CHECK(cudaMemcpy(d_lut, h_lut,
                           HISTOGRAM_SIZE * sizeof(unsigned char),
                           cudaMemcpyHostToDevice));

    // Pass 2: apply LUT
    applyLUT<<<grid1D, block1D>>>(d_blurred, d_heq, d_lut, numPixels);
    cudaEvent_t t5;
    CUDA_CHECK(cudaEventCreate(&t5));
    CUDA_CHECK(cudaEventRecord(t5));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Collect timing ────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventSynchronize(t5));
    float blurMs = 0, sobelMs = 0, heqMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&blurMs,  t0, t1));
    CUDA_CHECK(cudaEventElapsedTime(&sobelMs, t2, t3));
    CUDA_CHECK(cudaEventElapsedTime(&heqMs,   t4, t5));

    stats.blurTimeMs    = blurMs;
    stats.sobelTimeMs   = sobelMs;
    stats.histEqTimeMs  = heqMs;
    stats.totalGpuTimeMs = blurMs + sobelMs + heqMs;

    // ── Copy results back to host ──────────────────────────────────────────
    unsigned char* h_edges = (unsigned char*)malloc(imgBytes);
    unsigned char* h_heq   = (unsigned char*)malloc(imgBytes);

    CUDA_CHECK(cudaMemcpy(h_edges, d_edges, imgBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_heq,   d_heq,   imgBytes, cudaMemcpyDeviceToHost));

    // Compute mean output pixel value for the report
    double sumOut = 0.0;
    for (int i = 0; i < numPixels; i++) sumOut += h_heq[i];
    stats.meanPixelOut = (float)(sumOut / numPixels);

    // ── Save output images ─────────────────────────────────────────────────
    // Build output filenames:  <outputDir>/<stem>_edges.png  and  _heq.png
    string stem = outputPath;
    size_t lastDot = stem.rfind('.');
    if (lastDot != string::npos) stem = stem.substr(0, lastDot);

    string edgesPath = stem + "_edges.png";
    string heqPath   = stem + "_heq.png";

    stbi_write_png(edgesPath.c_str(), width, height, 1, h_edges, width);
    stbi_write_png(heqPath.c_str(),   width, height, 1, h_heq,   width);

    // ── Clean up ───────────────────────────────────────────────────────────
    free(h_edges);
    free(h_heq);
    stbi_image_free(h_input);
    cudaFree(d_src);
    cudaFree(d_blurred);
    cudaFree(d_edges);
    cudaFree(d_heq);
    cudaFree(d_lut);
    cudaFree(d_histogram);
    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaEventDestroy(t2);
    cudaEventDestroy(t3); cudaEventDestroy(t4); cudaEventDestroy(t5);

    return stats;
}

// ════════════════════════════════════════════════════════════════════════════
// HOST — Write per-image statistics CSV
// ════════════════════════════════════════════════════════════════════════════
__host__ void writeCsvReport(const string& csvPath,
                              const vector<ImageStats>& stats)
{
    ofstream csv(csvPath);
    if (!csv.is_open()) {
        fprintf(stderr, "Cannot write CSV: %s\n", csvPath.c_str());
        return;
    }

    // Header
    csv << "filename,width,height,blur_ms,sobel_ms,histeq_ms,"
        << "total_gpu_ms,mean_pixel_in,mean_pixel_out\n";

    for (const auto& s : stats) {
        csv << s.filename     << ","
            << s.width        << ","
            << s.height       << ","
            << s.blurTimeMs   << ","
            << s.sobelTimeMs  << ","
            << s.histEqTimeMs << ","
            << s.totalGpuTimeMs << ","
            << s.meanPixelIn  << ","
            << s.meanPixelOut << "\n";
    }

    csv.close();
    printf("CSV report written to: %s\n", csvPath.c_str());
}

// ════════════════════════════════════════════════════════════════════════════
// HOST — Print timing summary to stdout
// ════════════════════════════════════════════════════════════════════════════
__host__ void printSummary(const vector<ImageStats>& stats)
{
    if (stats.empty()) return;

    double totalBlur = 0, totalSobel = 0, totalHeq = 0, totalGpu = 0;
    int processed = 0;

    for (const auto& s : stats) {
        if (s.width == 0) continue;  // skipped / failed
        totalBlur  += s.blurTimeMs;
        totalSobel += s.sobelTimeMs;
        totalHeq   += s.histEqTimeMs;
        totalGpu   += s.totalGpuTimeMs;
        processed++;
    }

    printf("\n=== GPU Image Processing Summary ===\n");
    printf("Images processed      : %d\n",  processed);
    printf("Total GPU time        : %.3f ms\n", totalGpu);
    printf("  Gaussian blur       : %.3f ms\n", totalBlur);
    printf("  Sobel edge detect   : %.3f ms\n", totalSobel);
    printf("  Histogram equalise  : %.3f ms\n", totalHeq);
    if (processed > 0) {
        printf("Avg GPU time / image  : %.3f ms\n", totalGpu / processed);
    }
    printf("=====================================\n\n");
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN
// ════════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv)
{
    const string inputDir  = (argc > 1) ? argv[1] : "./images";
    const string outputDir = (argc > 2) ? argv[2] : "./output";
    const string csvPath   = (argc > 3) ? argv[3] : "./results.csv";

    // Create output directory if it doesn't exist
    mkdir(outputDir.c_str(), 0755);

    printf("Input directory  : %s\n", inputDir.c_str());
    printf("Output directory : %s\n", outputDir.c_str());

    // Discover all image files in the input directory
    vector<string> inputFiles = listImageFiles(inputDir);
    if (inputFiles.empty()) {
        fprintf(stderr, "No image files found in %s\n", inputDir.c_str());
        fprintf(stderr, "Run: python3 generate_images.py to create test images.\n");
        return 1;
    }

    printf("Found %zu image(s) to process.\n\n", inputFiles.size());

    vector<ImageStats> allStats;
    allStats.reserve(inputFiles.size());

    for (size_t i = 0; i < inputFiles.size(); i++) {
        const string& inPath = inputFiles[i];

        // Build the output path: <outputDir>/<input_filename>
        string basename = inPath.substr(inPath.rfind('/') + 1);
        string outPath  = outputDir + "/" + basename;

        printf("[%zu/%zu] Processing: %s\n", i+1, inputFiles.size(),
               basename.c_str());

        ImageStats s = processImage(inPath, outPath);
        if (s.width > 0) {
            printf("         %dx%d  blur=%.2fms  sobel=%.2fms  "
                   "histeq=%.2fms  total=%.2fms\n",
                   s.width, s.height,
                   s.blurTimeMs, s.sobelTimeMs,
                   s.histEqTimeMs, s.totalGpuTimeMs);
        }

        allStats.push_back(s);
    }

    printSummary(allStats);
    writeCsvReport(csvPath, allStats);

    return 0;
}

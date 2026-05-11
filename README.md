# GPU Image Processing Project
This project uses three CUDA GPU kernels to perform image processing using 200 synthetic portable grayMap (PGM) images.

## Setup

1. Run the `run.sh` script to execute the program.

### Step-by-step Setup
Manually execute the commands to below:
1. make clean
2. make generate_data
3. make build
4. make run 

#### Project Description

`Kernel 1 — gaussianBlur (constant memory)`
A 5×5 Gaussian filter with Pascal-triangle coefficients normalised to sum = 1 is stored in __constant__ memory (c_gaussKernel[25]). 
Every thread in a warp reads the same coefficient at the same time, which resolves as a single broadcast from the constant cache — no global memory traffic for the filter. 
Each thread computes one output pixel, clamping at image borders.

`Kernel 2 — sobelEdgeDetect (shared memory tiling)`
A (BLOCK_DIM+2) × (BLOCK_DIM+2) shared-memory tile is loaded cooperatively by the block, including a 1-pixel halo on all four sides. __syncthreads() ensures all halo data is visible before any thread applies the 3×3 Sobel operator. Without the tile, every thread would re-read the same global memory locations as its neighbours — the tile eliminates that redundancy entirely for interior pixels.

`Kernel 3 — buildHistogram + applyLUT (shared atomics + two-pass)`
Histogram equalisation requires a global 256-bin count, but thousands of threads atomically incrementing the same 256 global integers serialises badly. 
The solution: each block builds its own partial histogram in shared memory using atomicAdd (256 shared locations, very little contention), then merges its block-local result into the global array at the end. Pass 2 (applyLUT) maps each pixel through the precomputed equalisation LUT in a simple 1-D kernel — no branching, fully coalesced access.

##### Output

For each image the pipeline writes two processed PNGs (_edges.png, _heq.png) to ./output/ and records per-image timing and pixel statistics. After all images are processed, a results.csv is written and a timing summary is printed.
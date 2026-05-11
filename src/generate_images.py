#!/usr/bin/env python3
"""
generate_images.py

Generates NUM_IMAGES synthetic 256x256 grayscale PNG images in ./images/.
Each image has a different pattern so the GPU kernels process varied data.

Usage:
    python3 generate_images.py [num_images] [output_dir]

Defaults:
    num_images  = 200
    output_dir  = ./images
"""

import sys
import os
import math

# Use only stdlib so no pip install is needed; fall back to numpy/PIL if available
try:
    from PIL import Image
    import numpy as np
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

def make_image_raw(width, height, seed):
    """Generate a synthetic grayscale image as a flat list of [0,255] ints."""
    import random
    rng = random.Random(seed)
    pattern = seed % 6   # six different pattern types
    pixels = []

    for y in range(height):
        row = []
        for x in range(width):
            if pattern == 0:
                # Concentric circles
                cx, cy = width // 2, height // 2
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                v = int((math.sin(dist / 8.0) * 0.5 + 0.5) * 255)
            elif pattern == 1:
                # Diagonal gradient with noise
                v = int(((x + y) / (width + height)) * 200) + rng.randint(0, 55)
            elif pattern == 2:
                # Checkerboard with noise
                sq = 16
                base = 255 if ((x // sq + y // sq) % 2 == 0) else 0
                v = min(255, max(0, base + rng.randint(-30, 30)))
            elif pattern == 3:
                # Radial gradient
                cx, cy = width // 2, height // 2
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                maxd = math.sqrt(cx**2 + cy**2)
                v = int((1.0 - dist / maxd) * 255)
            elif pattern == 4:
                # Horizontal sine wave
                v = int((math.sin(x / 20.0 + seed / 10.0) * 0.5 + 0.5) * 255)
                v = min(255, max(0, v + rng.randint(-20, 20)))
            else:
                # Pure random noise
                v = rng.randint(0, 255)

            row.append(min(255, max(0, v)))
        pixels.extend(row)
    return pixels

def save_pgm(path, pixels, width, height):
    """Save a flat pixel list as a binary PGM file (no external deps needed)."""
    with open(path, 'wb') as f:
        header = f"P5\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))
        f.write(bytes(pixels))

def main():
    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out_dir    = sys.argv[2]      if len(sys.argv) > 2 else "./images"
    width, height = 256, 256

    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {num_images} synthetic {width}x{height} grayscale images "
          f"in '{out_dir}' ...")

    for i in range(num_images):
        pixels = make_image_raw(width, height, seed=i)
        filename = os.path.join(out_dir, f"synthetic_{i:04d}.pgm")

        if HAS_PILLOW:
            # Use Pillow if available for PNG output
            arr = np.array(pixels, dtype=np.uint8).reshape(height, width)
            img = Image.fromarray(arr, mode='L')
            png_path = os.path.join(out_dir, f"synthetic_{i:04d}.png")
            img.save(png_path)
        else:
            # Fall back to raw PGM (stb_image can read PGM files)
            save_pgm(filename, pixels, width, height)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{num_images} images ...")

    fmt = "PNG" if HAS_PILLOW else "PGM"
    print(f"Done. {num_images} {fmt} images written to '{out_dir}'.")
    print(f"\nNext step: make build && ./image_processing ./images ./output ./results.csv")

if __name__ == "__main__":
    main()
# Image Sampling and Quantization from Scratch

[![GitHub stars](https://img.shields.io/github/stars/NaiPondMa/Image_Process?style=social)](https://github.com/NaiPondMa/Image_Process/stargazers)

A small, educational Python project that demonstrates two foundations of digital
image processing: **spatial sampling** and **intensity quantization**. The core
resize and quantization operations are written as explicit pixel-by-pixel loops
so that the algorithms are easy to inspect, modify, and learn from.

If this repository helps you understand image processing, please
[give it a star](https://github.com/NaiPondMa/Image_Process). It helps other
learners discover the project too.

## What you will learn

A digital grayscale image can be viewed as a two-dimensional function:

```text
f(x, y) = brightness at spatial position (x, y)
```

Creating a digital image requires discretizing two different things:

- **Sampling** discretizes the spatial coordinates `(x, y)`. It determines how
  many pixels represent the image.
- **Quantization** discretizes the brightness value `f(x, y)`. It determines how
  many intensity levels each pixel may use.

They affect different parts of an image:

| Technique | Changes | Visible effect when reduced |
| --- | --- | --- |
| Sampling | Width and height | Loss of spatial detail and aliasing |
| Quantization | Number of intensity levels | Banding and loss of smooth gradients |

## 1. Image sampling

This project resizes grayscale images with **nearest-neighbor sampling**. For
each pixel in the output image, the algorithm finds the closest corresponding
pixel in the source image and copies its value.

For a scale factor `s`, the output dimensions are:

```text
new_height = original_height * s
new_width  = original_width  * s
```

The source coordinate for an output pixel `(i, j)` is calculated using inverse
mapping:

```text
source_row    = floor(i / s)
source_column = floor(j / s)
```

### Zooming

When `s > 1`, multiple output pixels can map to the same source pixel. The image
becomes larger, but no new detail is created. At high scale factors, square
pixel blocks become visible.

The example in `main.py` uses a zoom factor of `2.0`.

### Shrinking

To shrink by a factor `r`, the output dimensions are divided by `r`, and the
source coordinate is selected with:

```text
source_row    = floor(i * r)
source_column = floor(j * r)
```

Shrinking discards pixels. Nearest-neighbor sampling is intentionally simple,
but it may cause jagged edges or aliasing because it does not average the pixels
that were skipped.

The example in `main.py` uses a shrink factor of `4.0`.

## 2. Intensity quantization

An 8-bit grayscale image contains values from `0` (black) to `255` (white), for
a total of 256 possible intensities. Quantization reduces those intensities to
a smaller number of levels.

For a target bit depth `k`:

```text
number_of_levels = L = 2^k
bin_width = 256 / L
```

Each original pixel value `p` is first assigned to a level:

```text
level = floor(p / bin_width)
```

The level is then stretched back into the display range `0...255`:

```text
quantized_pixel = round(level * 255 / (L - 1))
```

This repository demonstrates:

| Bit depth | Available levels | Typical appearance |
| ---: | ---: | --- |
| 1-bit | 2 | Black and white |
| 3-bit | 8 | Strong visible intensity bands |
| 6-bit | 64 | Close to the original, with subtle loss |
| 8-bit | 256 | Original grayscale range |

Quantization changes pixel values but does not change the image width or
height. Fewer bits require less information per pixel, but also remove tonal
detail.

## Algorithm overview

```text
Color image
    |
    v
Grayscale image
    |--------------------------|
    v                          v
Nearest-neighbor sampling      Intensity quantization
    |                          |
    v                          v
Zoomed / shrunk image          1-bit / 3-bit / 6-bit image
```

OpenCV is used for image loading and color conversion, while the educational
sampling and quantization steps are implemented directly in Python loops.
Matplotlib displays the results.

## Project structure

```text
Image_Process/
|-- images/                          # Example input images
|-- scripts/
|   |-- sampling_component.py        # Manual zoom and shrink algorithms
|   `-- quantization_component.py    # Manual bit-depth quantization
|-- main.py                          # Runs and displays all demonstrations
|-- Dockerfile
`-- README.md
```

## Getting started

### Requirements

- Python 3.11 or newer
- NumPy
- OpenCV
- Matplotlib

### Installation

```bash
git clone https://github.com/NaiPondMa/Image_Process.git
cd Image_Process
python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install numpy opencv-python matplotlib
```

### Run the demonstration

In `main.py`, change the path passed to `cv2.imread(...)` so it points to one of
the images on your computer. For example:

```python
image = cv2.imread(r"images/cartoon_6a7499925e31c.jpg")
```

Then run:

```bash
python main.py
```

A Matplotlib window will compare the original image with its zoomed, shrunk,
1-bit, 3-bit, and 6-bit versions.

## Use the functions in your own code

```python
import cv2

from scripts.quantization_component import quantize_image
from scripts.sampling_component import (
    manual_shrinking_function,
    manual_zoom_function,
)

image = cv2.imread("images/flowers_6a74996980592.jpg")

zoomed = manual_zoom_function(image, 2.0)
shrunk = manual_shrinking_function(image, 4.0)
quantized = quantize_image(image, 3)
```

Valid quantization depths are from 1 to 8 bits. Zoom and shrink factors must be
greater than zero.

## Experiments to try

- Compare zoom factors such as `1.5`, `2.0`, and `4.0`.
- Compare shrinking before quantization with quantization before shrinking.
- Test quantization depths from 1 through 8 bits.
- Replace nearest-neighbor sampling with bilinear interpolation.
- Add an averaging filter before shrinking and observe how it reduces aliasing.
- Plot an intensity histogram before and after quantization.

## Current limitations

- Nearest-neighbor resizing favors clarity of implementation over visual
  quality.
- The algorithms use Python loops and are slower than optimized library
  implementations.
- Shrinking does not currently apply an anti-aliasing filter.
- The demonstration expects 8-bit images loaded in OpenCV's BGR format.

## Contributing

Issues, learning notes, documentation improvements, and algorithm additions are
welcome. If you add another technique, keep the implementation readable and
explain the underlying mathematics.

## Support the project

Found this explanation useful? Please
**[star the repository on GitHub](https://github.com/NaiPondMa/Image_Process)**
and share it with someone learning digital image processing.

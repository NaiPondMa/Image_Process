# Digital Image Processing: Algorithms & Coursework

[![GitHub stars](https://img.shields.io/github/stars/NaiPondMa/Image_Process?style=social)](https://github.com/NaiPondMa/Image_Process/stargazers)

A comprehensive Python repository implementing foundational and advanced digital image processing algorithms from scratch using NumPy, OpenCV, and Matplotlib.

> 📖 **Comprehensive Study & Lecture Material:**  
> For in-depth mathematical derivations, diagrams, and theoretical explanations across all quizzes, check out the **[`LECTURE_GUIDE.md`](LECTURE_GUIDE.md)**!

---

## Course Quizzes & Projects Overview

| Module | Topic | Core Techniques | Key Scripts & Components |
| :--- | :--- | :--- | :--- |
| **Quiz 1** | **Sampling & Quantization** | Spatial decimation & interpolation (Nearest-Neighbor, Bilinear), bit-depth reduction ($1, 3, 6, 8$-bit), false contouring analysis. | [`quiz1.py`](quiz1.py)<br>• [`scripts/sampling_component.py`](scripts/sampling_component.py)<br>• [`scripts/quantization_component.py`](scripts/quantization_component.py) |
| **Quiz 2** | **Intensity Transformations & Histograms** | Global vs. Local Histogram Equalization (LHE with $k_0, k_1, k_2$), Global vs. Local Power-Law (Gamma $\gamma$) correction. | [`quiz2.py`](quiz2.py)<br>• [`scripts/histogram_component.py`](scripts/histogram_component.py)<br>• [`scripts/gamma_compoent.py`](scripts/gamma_compoent.py) |
| **Quiz 3** | **Spatial Denoising & Sharpening** | Salt-and-pepper removal (Median $3 \times 3$), Gaussian noise filtering ($5 \times 5, \sigma=1$), Composite sharpening (Laplacian $\times$ Smoothed Sobel). | [`quiz3.py`](quiz3.py)<br>• [`scripts/filter_component.py`](scripts/filter_component.py)<br>• [`scripts/sharpening_component.py`](scripts/sharpening_component.py) |
| **Quiz 4** | **Frequency Domain & Noise Removal** | 2D Fourier Transform, Ideal/Notch LPF/HPF ($r=10, 50, 100$), Gaussian LPF/HPF ($D_0=10, 50, 100$), Gaussian Notch Reject Filter (GNRF). | [`quiz4.py`](quiz4.py)<br>• [`scripts/notch_filter_component.py`](scripts/notch_filter_component.py)<br>• [`scripts/gaussian_filter_component.py`](scripts/gaussian_filter_component.py) |
| **Term Project** | **Automated Lemon Counting** | Unified multi-stage pipeline: HSV color slicing, dual-band unripe lemon detection, morphological cleaning, distance transform + watershed. | [`term_project.py`](term_project.py)<br>• [`scripts/lemon_detector.py`](scripts/lemon_detector.py) |

---

## Repository Structure

```text
Image_Process/
|-- images/                          # Input datasets
|   |-- quiz3/                       # Blurry and noisy benchmark images
|   |-- quiz4/                       # Periodic noise & enhancement images
|   |-- Term-project/                # Lemon tree canopy images (1 to 4)
|   `-- result/                      # Exported output figures and diagnostic plots
|-- scripts/                         # Reusable processing components
|   |-- sampling_component.py        # Zooming & shrinking algorithms
|   |-- quantization_component.py    # Bit-depth quantization
|   |-- histogram_component.py       # Global & local histogram equalization
|   |-- gamma_compoent.py            # Global & block-wise gamma correction
|   |-- filter_component.py          # Median & Gaussian spatial smoothing
|   |-- sharpening_component.py      # Laplacian, Sobel & composite sharpening
|   |-- notch_filter_component.py    # Frequency ideal / notch filters
|   |-- gaussian_filter_component.py # Frequency Gaussian low/high/notch filters
|   `-- lemon_detector.py            # Unified end-to-end lemon detection engine
|-- quiz1.py                         # Quiz 1 driver (Sampling & Quantization)
|-- quiz2.py                         # Quiz 2 driver (Histogram & Gamma)
|-- quiz3.py                         # Quiz 3 driver (Spatial Denoising & Sharpening)
|-- quiz4.py                         # Quiz 4 driver (Frequency Filtering & Notch Rejection)
|-- term_project.py                  # Term Project driver (Lemon Counting)
|-- LECTURE_GUIDE.md                 # Complete student lecture & exam review notes
`-- README.md
```

---

## Detailed Quiz Breakdown

### Quiz 1: Sampling & Quantization
Demonstrates fundamental coordinate and amplitude discretization:
* **Sampling:** Zooming by $4.0\times$ and shrinking by $1/4\times$ via nearest-neighbor coordinate mapping.
* **Quantization:** Reducing 8-bit images down to 6-bit, 3-bit, and 1-bit representations. Highlights **false contouring** occurring at low bit depths.
* **Run:**
  ```powershell
  python quiz1.py
  ```

### Quiz 2: Intensity Transformations & Histogram Equalization
Enhances the tungsten filament image (`Filament.jpg`) to recover shadow details:
* **Global Histogram Equalization (GHE):** Over-amplifies background noise due to extreme contrast imbalance.
* **Local Histogram Equalization (LHE):** Evaluates local mean $m_{S_{xy}}$ and variance $\sigma_{S_{xy}}$ across $3 \times 3$, $7 \times 7$, and $11 \times 11$ windows with parameters $k_0, k_1, k_2$.
* **Gamma ($\gamma$) Correction:** Expands underexposed shadow detail with $\gamma < 1$ globally and over $2 \times 2$ / $3 \times 3$ subimage grids.
* **Run:**
  ```powershell
  python quiz2.py
  ```

### Quiz 3: Spatial Filtering — Noise Removal & Sharpening
Processes four corrupted benchmark images:
* **`noisy_cabin1_1.jpg` (Salt-and-Pepper):** Restored using a $3 \times 3$ **Median Filter**.
* **`noisy_cabin1_2.jpg` (Gaussian Noise):** Restored using a $5 \times 5$ **Gaussian Filter** ($\sigma = 1.0$).
* **`blurry_moon.jpg` & `blury_stop_sign.jpg`:** Sharpened on the luminance (Y) channel using a composite pipeline:
  $$\text{Sharpened } g = f + \left( \nabla^2 f \times \overline{|\nabla f|} \right) + \text{Contrast Stretch}$$
* **Run:**
  ```powershell
  python quiz3.py
  ```

### Quiz 4: Frequency Domain Filtering & Periodic Noise Removal
Demonstrates 2D Fourier filtering on `flower1.jpg` and `animals.jpg`:
* **Notch (Ideal) vs. Gaussian Filters:** Explains why ideal step cutoffs create circular **ringing (Gibbs phenomenon)**, while smooth Gaussian cutoffs eliminate ringing.
* **Periodic Noise Removal:** Uses **Gaussian Notch Reject Filters (GNRF)** to remove sinusoidal ripples:
  * Horizontal noise (`Noisy_flower1_horizontal`): Notches at $(dx=0, dy=\pm 69, \pm 138), D_0 = 6.0$.
  * Vertical noise (`Noisy_flower1_vertical`): Notches at $(dx=\pm 96, \pm 192, dy=0), D_0 = 6.0$.
  * Diagonal noise (`diag_noisy_animals`): Notches at $(\pm 28, \pm 37), (\pm 28, \pm 139), (\pm 28, \mp 65), D_0 = 5.0$.
* **Run:**
  ```powershell
  python quiz4.py
  ```

### Term Project: Automated Lemon Detection & Counting
Detects and counts ripe and unripe lemons in natural tree canopies:
* Unified pipeline: RGB $\rightarrow$ Gaussian smoothing $\rightarrow$ HSV color slicing $\rightarrow$ morphological opening/closing $\rightarrow$ Euclidean distance transform + watershed $\rightarrow$ circularity & area filtering.
* Results matching human ground truth:
  * `lemon_tree1`: 6 lemons (5 ripe + 1 unripe at bottom)
  * `lemon_tree2`: 27 lemons (>20 detected, resolving out-of-focus blur)
  * `lemon_tree3`: 13 lemons (Hue clamped to $H \le 26$, rejecting sunlit yellow leaves)
  * `lemon_tree4`: 122 lemons across wide canopy
* **Run:**
  ```powershell
  python term_project.py
  ```

---

## Technical Notes: Parameter Optimization FAQ

### How to Automatically Calculate the Best Gamma Value ($\gamma$)?
Instead of guessing $\gamma$, it can be computed automatically from the image's mean normalized brightness $\bar{I} \in (0, 1)$:
$$\gamma_{\text{auto}} = \frac{\log(0.5)}{\log(\bar{I})}$$
* For an underexposed dark image ($\bar{I} = 0.15$), $\gamma_{\text{auto}} \approx 0.36$, automatically brightening the shadows toward middle gray ($0.5$).
* For an overexposed bright image ($\bar{I} = 0.85$), $\gamma_{\text{auto}} \approx 4.25$, automatically dimming the highlights.

### How to Select $k$-Values in Local Histogram Enhancement?
For the condition $[m_{S_{xy}} \le k_0 M_G]$ and $[k_1 D_G \le \sigma_{S_{xy}} \le k_2 D_G]$:
* **$k_0 \approx 0.4 - 0.5$:** Restricts enhancement to the darkest $40-50\%$ of overall scene brightness.
* **$k_1 \approx 0.05 - 0.10$:** Rejects flat background regions to avoid boosting sensor noise.
* **$k_2 \approx 0.60 - 0.75$:** Prevents over-saturating regions that already have high local contrast.

---

## Getting Started

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

Install dependencies:
```bash
python -m pip install numpy opencv-python matplotlib
```

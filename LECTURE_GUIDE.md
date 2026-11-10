# Digital Image Processing: Complete Course Study Guide (Quiz 1 to Quiz 4)

This lecture guide breaks down the core concepts, mathematical principles, visual effects, and code implementations across all four quizzes in this repository.

---

## Quick Navigation
* [Quiz 1: Sampling & Intensity Quantization](#quiz-1-sampling--intensity-quantization)
* [Quiz 2: Spatial Intensity Transformations & Histogram Processing](#quiz-2-spatial-intensity-transformations--histogram-processing)
* [Quiz 3: Spatial Filtering — Noise Removal & Sharpening](#quiz-3-spatial-filtering--noise-removal--sharpening)
* [Quiz 4: Frequency Domain Filtering & Periodic Noise Removal](#quiz-4-frequency-domain-filtering--periodic-noise-removal)
* [Bonus: How to Optimize Gamma and k-Values](#bonus-how-to-optimize-gamma-and-k-values-quiz-2-faq)
* [Summary Comparison Cheatsheet](#summary-comparison-cheatsheet)

---

## Quiz 1: Sampling & Intensity Quantization

### 1. The Core Concept
A digital image is a 2D matrix of numbers $f[x, y]$ obtained by discretizing an analog visual scene:
* **Sampling:** Discretizes the continuous spatial coordinates $(x, y)$. Controls **Spatial Resolution** (number of rows $M$ and columns $N$).
* **Quantization:** Discretizes continuous light brightness into discrete levels. Controls **Intensity Resolution** (number of gray levels $L = 2^k$ for a $k$-bit image).

```text
Continuous World f(x, y) 
       │
       ▼ (Sampling: Spatial Grid M x N)
Grid Coordinates (x, y)
       │
       ▼ (Quantization: Amplitude 2^k levels)
Digital Matrix f[m, n] in range [0, 255]
```

### 2. Spatial Sampling: Zooming & Shrinking
* **Shrinking (Subsampling / Decimation):**
  Reduces image size by factor $S$. Nearest-neighbor decimation takes every $S$-th pixel:
  $$f_{\text{shrink}}[i, j] = f_{\text{orig}}[i \cdot S, j \cdot S]$$
  * *Artifact:* High-frequency details are skipped, causing **aliasing** (jagged staircases and Moiré patterns).
* **Zooming (Upsampling / Interpolation):**
  Increases image size by factor $Z$.
  * **Nearest Neighbor (Pixel Replication):** Copies the closest original pixel:
    $$f_{\text{zoom}}[x, y] = f_{\text{orig}}\left[\text{round}(x / Z), \text{round}(y / Z)\right]$$
    *Fast*, but creates obvious square pixel blocks at high zoom.
  * **Bilinear Interpolation:** Computes a weighted distance average of the 4 neighboring original pixels:
    $$f(x, y) = a x + b y + c x y + d$$
    Produces smooth, anti-aliased transitions without blockiness.

### 3. Intensity Quantization & False Contouring
An 8-bit image has $L = 256$ intensity levels ($0 = \text{black}$, $255 = \text{white}$). Reducing bit depth to $k$ bits yields $L_{\text{new}} = 2^k$ levels:
$$\text{Step Size } \Delta = \frac{256}{2^k}$$
$$q[x, y] = \text{round}\left( \frac{\lfloor f[x, y] / \Delta \rfloor \cdot 255}{2^k - 1} \right)$$

* **8-bit (256 levels):** Smooth, continuous photographic tone.
* **6-bit (64 levels):** Minor loss, virtually indistinguishable to the human eye.
* **3-bit (8 levels):** **False Contouring** appears. The human eye perceives artificial bands/ridges (Mach bands) where smooth gradients used to be.
* **1-bit (2 levels):** Pure binary silhouette (only 0 or 255). All midtones and shading are lost.

**Associated Code:**
* Driver: [`quiz1.py`](quiz1.py)
* Components: [`scripts/sampling_component.py`](scripts/sampling_component.py), [`scripts/quantization_component.py`](scripts/quantization_component.py)

---

## Quiz 2: Spatial Intensity Transformations & Histogram Processing

### 1. The Core Concept
Point operations apply a function $s = T(r)$ to individual pixels $r$ without checking neighbor pixels. Used to balance lighting, expand contrast, and reveal details in shadows or glare.

### 2. Histogram Equalization: Global vs. Local
The histogram $p_r(r_k) = \frac{n_k}{M \cdot N}$ represents the probability of intensity $r_k$.

#### Global Histogram Equalization (GHE)
Transforms intensities using the normalized Cumulative Distribution Function (CDF):
$$s_k = T(r_k) = (L - 1) \sum_{j=0}^{k} p_r(r_j)$$
* **Advantage:** Completely automatic; flattens the histogram to maximize overall dynamic contrast across $[0, 255]$.
* **Limitation:** In images with a large dark background and a small bright foreground (like `Filament.jpg`), GHE over-amplifies background noise and washes out the fine details of the filament.

#### Local Histogram Equalization (LHE) / Statistical Local Enhancement
Calculates local mean $m_{S_{xy}}$ and variance $\sigma_{S_{xy}}^2$ in an $n \times n$ neighborhood:
$$g(x, y) = \begin{cases} 
E \cdot f(x, y) & \text{if } [m_{S_{xy}} \le k_0 M_G] \text{ and } [k_1 D_G \le \sigma_{S_{xy}} \le k_2 D_G] \\
f(x, y) & \text{otherwise}
\end{cases}$$
* $M_G, D_G$: Global mean and standard deviation.
* $k_0$: Maximum brightness ratio for enhancement (limits brightening to dark regions).
* $k_1, k_2$: Variance bounds (ensures enhancement only occurs where edges/textures exist, avoiding uniform noise).
* $E$: Local gain multiplier ($E > 1$).

**Neighborhood Window Comparison:**
* $3 \times 3$: Highly localized, introduces granular noise.
* $7 \times 7$: Optimal balance; brings out hidden filament loops cleanly.
* $11 \times 11$: Coarser smoothing, closer to macro contrast.

### 3. Power-Law (Gamma $\gamma$) Transformation
$$s = c \cdot r^\gamma \quad (r, s \in [0, 1])$$
* **$\gamma < 1$ (e.g., $\gamma = 0.4$):** Stretches dark shadows into visible midtones; compresses bright highlights. Perfect for underexposed pictures.
* **$\gamma > 1$ (e.g., $\gamma = 2.5$):** Darkens midtones; stretches highlight contrast. Perfect for washed-out overexposed images.
* **Local Gamma ($2 \times 2$ or $3 \times 3$ grid):** Dark quadrants get low $\gamma \approx 0.4$, while already-bright quadrants keep $\gamma \approx 1.0$, preventing overexposure.

**Associated Code:**
* Driver: [`quiz2.py`](quiz2.py)
* Components: [`scripts/histogram_component.py`](scripts/histogram_component.py), [`scripts/gamma_compoent.py`](scripts/gamma_compoent.py)

---

## Quiz 3: Spatial Filtering — Noise Removal & Sharpening

### 1. The Core Concept
Spatial filtering slides an $m \times n$ mask (kernel) $w$ over the image to perform neighborhood operations:
* **Smoothing (Low-pass):** Averages neighboring pixels to attenuate noise.
* **Sharpening (High-pass):** Calculates spatial derivatives to enhance edges and fine details.

### 2. Noise Removal: Choosing the Right Filter

| Noise Type | Characteristic Appearance | Best Spatial Filter | Why Other Filters Fail |
| :--- | :--- | :--- | :--- |
| **Salt-and-Pepper** | Isolated white (255) and black (0) specks | **Median Filter ($3 \times 3$)** | Linear box/Gaussian filters average the extreme 0 and 255 values into neighbors, turning sharp noise into blurry gray smudges. Median replaces the outlier with the center rank value. |
| **Gaussian Noise** | Fine grain variations across all pixels | **Gaussian Filter ($5 \times 5, \sigma = 1.0$)** | Averages random fluctuations using bell-curve spatial weights, preserving edge geometry better than an unweighted box filter. |

### 3. Blurry Image Sharpening: Composite Multi-Stage Method
Sharpening with a raw 2nd-derivative Laplacian also amplifies background noise. The composite pipeline solves this:

$$\text{Final Enhanced } g = f + \left( \nabla^2 f \times \overline{|\nabla f|} \right) + \text{Contrast Stretch}$$

1. **Laplacian ($\nabla^2 f$):** Highlights fine lines and point discontinuities.
2. **Sobel Gradient ($|\nabla f| \approx |G_x| + |G_y|$):** Computes strong 1st-derivative macro edges.
3. **Smoothed Sobel ($\overline{|\nabla f|}$):** $5 \times 5$ averaging removes gradient noise.
4. **Masked Product ($\nabla^2 f \times \overline{|\nabla f|}$):** Gates the enhancement — strong boost at true edges, near-zero boost in flat noisy areas!
5. **Contrast Stretch:** Scales the dynamic range to make the sharpened image pop.

**Associated Code:**
* Driver: [`quiz3.py`](quiz3.py)
* Components: [`scripts/filter_component.py`](scripts/filter_component.py), [`scripts/sharpening_component.py`](scripts/sharpening_component.py)

---

## Quiz 4: Frequency Domain Filtering & Periodic Noise Removal

### 1. The Core Concept
Using the Fourier Convolution Theorem, slow spatial convolution becomes fast pointwise multiplication in frequency space:
$$f(x, y) * h(x, y) \iff F(u, v) \cdot H(u, v)$$

* Low frequencies (center of shifted FFT) represent smooth background shading.
* High frequencies (away from center) represent edges, textures, and fine noise.

### 2. Notch (Ideal) vs. Gaussian Filters

```text
    Ideal / Notch Cutoff                       Gaussian Cutoff
    H(u, v)                                    H(u, v)
    1.0 ┌──────────┐                           1.0 ┌────\
        │          │                               │     \
        │          │                               │      \
    0.0 └───┴──────┴─────► Radius D            0.0 └───┴───\──► Radius D
            r                                          D0
    [ Sharp step in frequency ]                [ Smooth exponential curve ]
    --> Sinc ripples in space                  --> Pure Gaussian in space
    --> Severe Gibbs Ringing                   --> Zero Ringing
```

* **Ideal / Notch Filters:**
  * Low-pass: $H(u, v) = 1$ if $D(u, v) \le r$, else $0$.
  * High-pass: $H(u, v) = 0$ if $D(u, v) \le r$, else $1$.
  * *Artifact:* The inverse Fourier transform of a sharp step is a sinc function $\frac{\sin(x)}{x}$. Its ripples cause **concentric ringing halos** around all image edges.
* **Gaussian Filters:**
  * Low-pass: $H(u, v) = e^{-D^2(u, v) / (2 D_0^2)}$.
  * High-pass: $H(u, v) = 1 - e^{-D^2(u, v) / (2 D_0^2)}$.
  * *Advantage:* The Fourier transform of a Gaussian is another Gaussian — **completely eliminates ringing**.

### 3. Periodic Noise & Notch Reject Filtering
Periodic sinusoidal interference $n(x, y) = A \sin(2\pi u_0 x + 2\pi v_0 y)$ manifests as **isolated symmetric delta spikes** in Fourier space:
* **Horizontal noise lines:** Vary vertically $\implies$ Spikes lie on the **vertical axis** ($dx = 0, dy = \pm v_0$).
* **Vertical noise lines:** Vary horizontally $\implies$ Spikes lie on the **horizontal axis** ($dx = \pm u_0, dy = 0$).
* **Diagonal noise lines:** Vary diagonally $\implies$ Spikes lie **off-axis** at symmetric pairs $(\pm dx, \pm dy)$.

#### Gaussian Notch Reject Filter (GNRF)
Attenuates the noise spikes without affecting the DC component at the center $(0, 0)$:
$$H_{\text{GNR}}(u, v) = \prod_{k=1}^Q \left( 1 - e^{-\frac{D_k^2(u, v)}{2 D_0^2}} \right) \left( 1 - e^{-\frac{D_{-k}^2(u, v)}{2 D_0^2}} \right)$$

**Optimal Parameters for Quiz 4 Benchmark Images:**
1. **`Noisy_flower1_horizontal.jpg`:** Spikes at $(dx=0, dy=\pm 69)$ and harmonic $(0, \pm 138)$, $D_0 = 6.0$.
2. **`Noisy_flower1_vertical.jpg`:** Spikes at $(dx=\pm 96, dy=0)$ and harmonic $(\pm 192, 0)$, $D_0 = 6.0$.
3. **`diag_noisy_animals.jpg`:** Spikes at $(\pm 28, \pm 37)$, $(\pm 28, \pm 139)$, $(\pm 28, \mp 65)$, $D_0 = 5.0$.

**Associated Code:**
* Driver: [`quiz4.py`](quiz4.py)
* Components: [`scripts/notch_filter_component.py`](scripts/notch_filter_component.py), [`scripts/gaussian_filter_component.py`](scripts/gaussian_filter_component.py)

---

## Bonus: How to Optimize Gamma and k-Values (Quiz 2 FAQ)

In `README.md`, you asked:
> *"Problem we found is calculating or finding the best gamma value and k-values... manually input or giving a function to calculate best optimize?"*

Here are the standard automated methods used in professional image processing:

### 1. Automated Gamma Optimization
Instead of guessing $\gamma$, calculate it automatically from the image's mean brightness $\bar{I} \in [0, 1]$:

$$\gamma_{\text{auto}} = \frac{\log(0.5)}{\log(\bar{I})}$$

* **Why this works:** If the image is dark (e.g. $\bar{I} = 0.15$), $\gamma = \frac{\log(0.5)}{\log(0.15)} \approx 0.36$, which automatically brightens the image so its average becomes $0.5$ (middle gray).
* If the image is overexposed ($\bar{I} = 0.85$), $\gamma \approx 4.25$, which automatically darkens it.

```python
def auto_gamma(image, target_mean=0.5):
    norm = image.astype(np.float64) / 255.0
    mean_val = np.mean(norm)
    if mean_val <= 0 or mean_val >= 1:
        return 1.0
    gamma = np.log(target_mean) / np.log(mean_val)
    return np.clip(gamma, 0.2, 5.0)
```

### 2. Automated $k$-Value Selection for Local Histogram Enhancement
For the condition $[m_{S_{xy}} \le k_0 M_G]$ and $[k_1 D_G \le \sigma_{S_{xy}} \le k_2 D_G]$:
* **$k_0$ (Shadow Gate):** Set $k_0 \approx 0.4 - 0.5$ (ensures enhancement only activates in the bottom $40-50\%$ of overall brightness).
* **$k_1$ (Noise Floor):** Set $k_1 \approx 0.05 - 0.10$ (rejects flat background noise where variance is near zero).
* **$k_2$ (Edge Ceiling):** Set $k_2 \approx 0.60 - 0.75$ (prevents over-enhancing regions that already have high natural contrast).

---

## Summary Comparison Cheatsheet

| Task / Problem | Domain | Recommended Method | Key Formulas & Parameters |
| :--- | :--- | :--- | :--- |
| **Resize / Interpolate** | Spatial | Bilinear Interpolation | $f(x,y) = ax + by + cxy + d$ |
| **Bit-Depth Reduction** | Spatial | Quantization Step Division | Step $\Delta = 256 / 2^k$ |
| **Global Contrast Boost** | Spatial | Histogram Equalization | $s_k = (L-1)\sum p_r(r_j)$ |
| **Reveal Dark Shadows** | Spatial | Local Hist Eq / Gamma $< 1$ | Window $7 \times 7, k_0=0.4$ / $\gamma = 0.4$ |
| **Salt-and-Pepper Noise** | Spatial | Median Filter | $3 \times 3$ sliding window rank |
| **Gaussian Random Noise** | Spatial | Gaussian Spatial Filter | $5 \times 5, \sigma = 1.0$ |
| **Blurry Edge Sharpening** | Spatial | Laplacian $\times$ Smoothed Sobel | $f + (\nabla^2 f \cdot \overline{|\nabla f|})$ |
| **Frequency Smoothing** | Frequency | Gaussian Low-Pass Filter | $H = e^{-D^2 / (2D_0^2)}$ (no ringing) |
| **Frequency Edge Extraction**| Frequency | Gaussian High-Pass Filter | $H = 1 - e^{-D^2 / (2D_0^2)}$ |
| **Periodic Sinusoidal Noise** | Frequency | Gaussian Notch Reject Filter | Notches at $(\pm dx, \pm dy), D_0 \approx 5-6$ |

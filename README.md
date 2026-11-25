<div align="center">

# 🔬 Vision Lab: Harris Corners & Fourier Image Analysis

**An interactive desktop tool for exploring classic computer-vision building blocks: Harris corner detection from first principles, 2D DFT magnitude and phase spectra, and image reconstruction that mixes one image's magnitude with another's phase.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

</div>

---

## 📖 Overview

Built for the **Image Processing & Computer Vision** course at Cairo University. Rather than calling `cv2.cornerHarris`, the detector is implemented step by step, so every parameter can be tuned and its effect seen immediately.

## ✨ Features

### 1️⃣ Harris Corner Detection
- Image gradients (`Ix`, `Iy`) computed with **Sobel** operators
- Structure tensor `M` smoothed with a **Gaussian** window (σ = 1)
- Corner response **R = det(M) − k·trace(M)²** with live sliders for **k** and the **threshold**
- Vectorised thresholding, with detected corners overlaid on the image and counted
- **Zoom and pan** (drag or scroll) for inspecting individual corners

### 2️⃣ DFT Analysis
- Centred 2D DFT (`fftshift`) of two images side by side
- Switch between **magnitude** (log-scaled) and **phase** spectra

### 3️⃣ Magnitude / Phase Reconstruction
- Choose which image supplies the **magnitude** and which supplies the **phase**
- Weight each component with percentage sliders
- Inverse DFT shows the classic result: **phase carries most of the structure**

## 🧮 The Math

$$M = \sum_{w} G_\sigma * \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}, \qquad R = \det(M) - k\,\operatorname{tr}(M)^2$$

$$F_{mix}(u,v) = \alpha\,|F_1(u,v)|\; e^{\,j\,\beta\,\angle F_2(u,v)}$$

## 🚀 Getting Started

```bash
git clone https://github.com/Mohamedd-Ayyman/vision-lab.git
cd vision-lab
pip install -r requirements.txt
python vision_lab.py
```

## 👤 Author

**Mohamed Ayman** · [GitHub](https://github.com/Mohamedd-Ayyman) · [LinkedIn](https://linkedin.com/in/mohamed-ayman-8787b0269)

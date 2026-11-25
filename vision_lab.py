"""Vision Lab: interactive Harris corner detection, DFT analysis and
magnitude/phase image reconstruction (Tkinter + OpenCV + Matplotlib)."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
import os
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageProcessor:
    def __init__(self):
        self.img_color = None
        self.img_gray = None
        self.img_float = None
        self.image_path = None
        self.img1_color = None
        self.img2_color = None
        self.img1_gray = None
        self.img2_gray = None
        self.sigma = 1.0
        self.k = 0.05
        self.threshold = 0.001
        self.corner_radius = 3
        self.display_mode = 'magnitude'
        self.zoom_center = None
        self.zoom_start = None
        self.zoom_base = 1.0
        self.zoom_level = 1.0
        self.harris_step_images = []
        self.mag1 = None
        self.pha1 = None
        self.mag2 = None
        self.pha2 = None
        self.dft_shift1 = None
        self.dft_shift2 = None
        self.mag_percent = 100
        self.phase_percent = 100
        self.use_mag_from = 1
        self.use_phase_from = 2
        self.setup_ui()

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("Vision Lab: Harris Corners & Fourier Analysis")
        self.root.geometry("1800x1200")
        self.root.configure(bg="#f0f0f0")

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10, "bold"), padding=5)
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 10))
        style.configure("TNotebook", background="#f0f0f0")
        style.configure("TNotebook.Tab", padding=[
                        12, 6], font=("Arial", 10, "bold"))
        style.configure("TScale", background="#f0f0f0")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Harris Corner Detection")
        self.notebook.add(self.tab2, text="DFT Analysis")
        self.notebook.add(self.tab3, text="Image Reconstruction")

        self.setup_harris_tab()
        self.setup_dft_tab()
        self.setup_recon_tab()

    def setup_harris_tab(self):
        top_controls = ttk.Frame(self.tab1)
        top_controls.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(top_controls, text="Upload Image", style="TButton",
                   command=self.upload_harris_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_controls, text="Reset Zoom", style="TButton",
                   command=self.reset_zoom).pack(side=tk.LEFT, padx=5)

        controls_frame = ttk.LabelFrame(self.tab1, text="Controls")
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        row1 = ttk.Frame(controls_frame)
        row1.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(row1, text="k:").pack(side=tk.LEFT, padx=5)
        self.k_slider = ttk.Scale(row1, from_=0.04, to=0.06, length=200,
                                  command=self.update_harris_params)
        self.k_slider.set(self.k)
        self.k_slider.pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(controls_frame)
        row2.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(row2, text="Threshold:").pack(side=tk.LEFT, padx=5)
        self.threshold_slider = ttk.Scale(row2, from_=0.0001, to=0.01, length=200,
                                          command=self.update_harris_params)

        self.threshold_slider.set(self.threshold)
        self.threshold_slider.pack(side=tk.LEFT, padx=5)

        self.fig_harris, self.axs_harris = plt.subplots(1, 2, figsize=(40, 20))
        self.canvas_harris = FigureCanvasTkAgg(self.fig_harris, self.tab1)
        self.canvas_harris.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        for ax in self.axs_harris:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Upload an image',
                    horizontalalignment='center', verticalalignment='center')

        self.canvas_harris.mpl_connect(
            'button_press_event', self.on_click_harris)
        self.canvas_harris.mpl_connect('scroll_event', self.on_scroll)
        self.canvas_harris.mpl_connect('button_press_event', self.start_zoom)
        self.canvas_harris.mpl_connect('button_release_event', self.end_zoom)
        self.canvas_harris.mpl_connect('motion_notify_event', self.do_zoom)

    def setup_dft_tab(self):
        controls_frame = ttk.LabelFrame(self.tab2, text="Controls")
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        row1 = ttk.Frame(controls_frame)
        row1.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(row1, text="Upload Image 1", style="TButton",
                   command=lambda: self.upload_and_update_dft(1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="Upload Image 2", style="TButton",
                   command=lambda: self.upload_and_update_dft(2)).pack(side=tk.LEFT, padx=5)

        display_frame = ttk.Frame(controls_frame)
        display_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(display_frame, text="Display:").pack(side=tk.LEFT, padx=5)
        self.display_var = tk.StringVar(value="magnitude")
        ttk.Radiobutton(display_frame, text="Magnitude", variable=self.display_var,
                        value="magnitude", command=self.update_display_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(display_frame, text="Phase", variable=self.display_var,
                        value="phase", command=self.update_display_mode).pack(side=tk.LEFT, padx=5)

        self.fig_dft, self.axs_dft = plt.subplots(2, 2, figsize=(20, 12))
        self.canvas_dft = FigureCanvasTkAgg(self.fig_dft, self.tab2)
        self.canvas_dft.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        for ax in self.axs_dft.flatten():
            ax.axis('off')
            ax.text(0.5, 0.5, 'Upload images to begin',
                    horizontalalignment='center', verticalalignment='center')

    def setup_recon_tab(self):
        controls_frame = ttk.LabelFrame(self.tab3, text="Controls")
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        upload_frame = ttk.Frame(controls_frame)
        upload_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(upload_frame, text="Upload Image 1", style="TButton",
                   command=lambda: self.upload_and_update_dft(1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(upload_frame, text="Upload Image 2", style="TButton",
                   command=lambda: self.upload_and_update_dft(2)).pack(side=tk.LEFT, padx=5)

        source_frame = ttk.Frame(controls_frame)
        source_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(source_frame, text="Magnitude Source:").pack(
            side=tk.LEFT, padx=5)
        self.mag_source_var = tk.IntVar(value=1)
        ttk.Radiobutton(source_frame, text="Image 1", variable=self.mag_source_var,
                        value=1, command=self.update_image_sources).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(source_frame, text="Image 2", variable=self.mag_source_var,
                        value=2, command=self.update_image_sources).pack(side=tk.LEFT, padx=5)

        ttk.Label(source_frame, text="Phase Source:").pack(
            side=tk.LEFT, padx=20)
        self.phase_source_var = tk.IntVar(value=2)
        ttk.Radiobutton(source_frame, text="Image 1", variable=self.phase_source_var,
                        value=1, command=self.update_image_sources).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(source_frame, text="Image 2", variable=self.phase_source_var,
                        value=2, command=self.update_image_sources).pack(side=tk.LEFT, padx=5)

        recon_type_frame = ttk.Frame(controls_frame)
        recon_type_frame.pack(fill=tk.X, padx=5, pady=5)

        slider_frame = ttk.Frame(controls_frame)
        slider_frame.pack(fill=tk.X, padx=5, pady=5)

        img1_mag_frame = ttk.Frame(slider_frame)
        img1_mag_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(img1_mag_frame, text="Magnitude (%):").pack(
            side=tk.LEFT, padx=5)
        self.mag_slider = ttk.Scale(img1_mag_frame, from_=0, to=100, length=300,
                                    command=self.update_recon_params)
        self.mag_slider.set(self.mag_percent)
        self.mag_slider.pack(side=tk.LEFT, padx=5)
        self.mag_label = ttk.Label(img1_mag_frame, text="100%")
        self.mag_label.pack(side=tk.LEFT, padx=5)

        img2_phase_frame = ttk.Frame(slider_frame)
        img2_phase_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(img2_phase_frame, text="Phase (%):").pack(
            side=tk.LEFT, padx=5)
        self.phase_slider = ttk.Scale(img2_phase_frame, from_=0, to=100, length=300,
                                      command=self.update_recon_params)
        self.phase_slider.set(self.phase_percent)
        self.phase_slider.pack(side=tk.LEFT, padx=5)
        self.phase_label = ttk.Label(img2_phase_frame, text="100%")
        self.phase_label.pack(side=tk.LEFT, padx=5)

        self.fig_recon = plt.figure(figsize=(16, 12))
        self.ax_recon = self.fig_recon.add_subplot(111)
        self.ax_recon.axis('off')
        self.ax_recon.text(0.5, 0.5, 'Upload images to create reconstruction',
                           horizontalalignment='center', verticalalignment='center')
        self.canvas_recon = FigureCanvasTkAgg(self.fig_recon, self.tab3)
        self.canvas_recon.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_image_sources(self):
        self.use_mag_from = self.mag_source_var.get()
        self.use_phase_from = self.phase_source_var.get()
        self.update_recon_params()

    def on_click_harris(self, event):
        if event.inaxes == self.axs_harris[1] and hasattr(self, 'img_corners'):
            self.zoom_center = (int(event.xdata), int(event.ydata))
            if self.zoom_level > 1:
                self.update_zoom()

    def start_zoom(self, event):
        if event.button == 3:
            self.zoom_start = (event.x, event.y)

    def end_zoom(self, event):
        self.zoom_start = None

    def do_zoom(self, event):
        if self.zoom_start and event.button == 3:
            dy = self.zoom_start[1] - event.y
            self.zoom_level = max(1, min(10, self.zoom_level + dy * 0.01))
            self.update_zoom()

    def on_scroll(self, event):
        if event.inaxes == self.axs_harris[1] and hasattr(self, 'img_corners'):
            self.zoom_center = (int(event.xdata), int(event.ydata))
            self.zoom_level = max(1, min(10, self.zoom_level + event.step))
            self.update_zoom()

    def reset_zoom(self):
        self.zoom_center = None
        self.zoom_level = 1
        self.update_zoom()

    def update_harris_response(self):
        if self.img_float is None:
            return

        Ix = sobel(self.img_float, axis=1)
        Iy = sobel(self.img_float, axis=0)

        Ix2 = gaussian_filter(Ix * Ix, self.sigma)
        Iy2 = gaussian_filter(Iy * Iy, self.sigma)
        Ixy = gaussian_filter(Ix * Iy, self.sigma)

        det_M = Ix2 * Iy2 - Ixy**2
        trace_M = Ix2 + Iy2
        self.R = det_M - self.k * (trace_M**2)

        threshold = self.threshold * self.R.max()
        self.corner_points = []
        self.img_corners = np.copy(self.img_color)

        # Vectorised thresholding (replaces a per-pixel Python loop)
        ys, xs = np.nonzero(self.R > threshold)
        self.corner_points = list(zip(xs.tolist(), ys.tolist()))
        for x, y in self.corner_points:
            cv2.circle(self.img_corners, (x, y),
                       self.corner_radius, (0, 0, 255), 1)

        self.edges = np.sqrt(Ix**2 + Iy**2) / np.max(np.sqrt(Ix**2 + Iy**2))

        self.axs_harris[1].clear()
        self.axs_harris[1].imshow(cv2.cvtColor(
            self.img_corners, cv2.COLOR_BGR2RGB))
        self.axs_harris[1].set_title(
            f'Corners Detected ({len(self.corner_points)})')
        self.axs_harris[1].axis('off')

        self.canvas_harris.draw()

    def update_zoom(self):
        if hasattr(self, 'corner_points') and len(self.corner_points) > 0:
            zoom_level = int(self.zoom_level)
            self.axs_harris[1].clear()

            if zoom_level == 1:
                self.axs_harris[1].imshow(cv2.cvtColor(
                    self.img_corners, cv2.COLOR_BGR2RGB))
                self.axs_harris[1].set_title(
                    f'Corners Detected ({len(self.corner_points)})')
                self.zoom_center = None
            else:
                height, width = self.img_corners.shape[:2]
                window_size = min(width, height) // zoom_level

                if self.zoom_center is None and len(self.corner_points) > 0:
                    self.zoom_center = self.corner_points[0]

                if self.zoom_center:
                    center_x, center_y = self.zoom_center
                    x_min = max(0, center_x - window_size//2)
                    y_min = max(0, center_y - window_size//2)
                    x_max = min(width, x_min + window_size)
                    y_max = min(height, y_min + window_size)

                    if x_max == width:
                        x_min = max(0, width - window_size)
                    if y_max == height:
                        y_min = max(0, height - window_size)

                    self.axs_harris[1].imshow(cv2.cvtColor(
                        self.img_corners[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2RGB))
                    self.axs_harris[1].set_title(
                        f'Corners Detected (Zoom: {zoom_level}x)')

            self.axs_harris[1].axis('off')
            self.canvas_harris.draw()

    def update_harris_params(self, val=None):
        self.k = float(self.k_slider.get())
        self.threshold = float(self.threshold_slider.get())
        self.update_harris_response()
        self.update_zoom()

    def upload_harris_image(self):
        file_path = filedialog.askopenfilename(
            title='Select Image',
            filetypes=[
                ('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')]
        )

        if file_path:
            self.img_color = cv2.imread(file_path)
            if self.img_color is not None:
                self.img_gray = cv2.cvtColor(
                    self.img_color, cv2.COLOR_BGR2GRAY)
                self.img_float = self.img_gray.astype(np.float64) / 255.0
                self.image_path = file_path

                self.axs_harris[0].clear()
                self.axs_harris[0].imshow(cv2.cvtColor(
                    self.img_color, cv2.COLOR_BGR2RGB))
                self.axs_harris[0].set_title('Original Image')
                self.axs_harris[0].axis('off')

                self.update_harris_response()
                self.update_zoom()

    def upload_dft_image(self, img_num):
        file_path = filedialog.askopenfilename(
            title=f'Select Image {img_num}',
            filetypes=[
                ('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')]
        )

        if file_path:
            img_color = cv2.imread(file_path)
            if img_color is not None:
                img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

                if img_num == 1:
                    self.img1_color = img_color
                    self.img1_gray = img_gray
                else:
                    self.img2_color = img_color
                    self.img2_gray = img_gray

                if self.img1_gray is not None and self.img2_gray is not None:
                    h1, w1 = self.img1_gray.shape
                    h2, w2 = self.img2_gray.shape

                    min_h = min(h1, h2)
                    min_w = min(w1, w2)

                    self.img1_gray = cv2.resize(self.img1_gray, (min_w, min_h))
                    self.img2_gray = cv2.resize(self.img2_gray, (min_w, min_h))

                return True
        return False

    def update_recon_params(self, val=None):
        if self.img1_gray is None or self.img2_gray is None:
            return

        self.mag_percent = int(self.mag_slider.get())
        self.phase_percent = int(self.phase_slider.get())

        self.mag_label.config(text=f"{self.mag_percent}%")
        self.phase_label.config(text=f"{self.phase_percent}%")

        self.create_mixed_image()

    def create_mixed_image(self):
        if self.img1_gray is None or self.img2_gray is None:
            return

        dft1 = cv2.dft(np.float32(self.img1_gray),
                       flags=cv2.DFT_COMPLEX_OUTPUT)
        dft2 = cv2.dft(np.float32(self.img2_gray),
                       flags=cv2.DFT_COMPLEX_OUTPUT)

        self.dft_shift1 = np.fft.fftshift(dft1)
        self.dft_shift2 = np.fft.fftshift(dft2)

        self.mag1 = cv2.magnitude(
            self.dft_shift1[:, :, 0], self.dft_shift1[:, :, 1])
        self.pha1 = np.angle(
            self.dft_shift1[:, :, 0] + 1j * self.dft_shift1[:, :, 1])

        self.mag2 = cv2.magnitude(
            self.dft_shift2[:, :, 0], self.dft_shift2[:, :, 1])
        self.pha2 = np.angle(
            self.dft_shift2[:, :, 0] + 1j * self.dft_shift2[:, :, 1])

        source_mag = self.mag1 if self.use_mag_from == 1 else self.mag2
        source_phase = self.pha1 if self.use_phase_from == 1 else self.pha2

        min_rows = min(source_mag.shape[0], source_phase.shape[0])
        min_cols = min(source_mag.shape[1], source_phase.shape[1])

        source_mag_cropped = source_mag[:min_rows, :min_cols]
        source_phase_cropped = source_phase[:min_rows, :min_cols]

        mag_factor = self.mag_percent / 100.0
        pha_factor = self.phase_percent / 100.0

        real = source_mag_cropped * mag_factor * \
            np.cos(source_phase_cropped * pha_factor)
        imaginary = source_mag_cropped * mag_factor * \
            np.sin(source_phase_cropped * pha_factor)
        mixed_complex = real + 1j * imaginary

        mixed_shift_back = np.fft.ifftshift(mixed_complex)
        mixed_idft = np.fft.ifft2(mixed_shift_back)
        mixed_image = np.abs(mixed_idft)

        mixed_norm = cv2.normalize(mixed_image, None, 0, 255, cv2.NORM_MINMAX)
        self.mixed_img = np.uint8(mixed_norm)

        color_mapping = None
        if self.mag_percent > 0 and self.phase_percent > 0:
            color_mapping = cv2.COLORMAP_VIRIDIS
        elif self.mag_percent > 0:
            color_mapping = cv2.COLORMAP_BONE
        else:
            color_mapping = cv2.COLORMAP_TWILIGHT

        if color_mapping:
            colored_result = cv2.applyColorMap(self.mixed_img, color_mapping)
            self.mixed_img = colored_result

        self.ax_recon.clear()
        if len(self.mixed_img.shape) == 3:
            self.ax_recon.imshow(cv2.cvtColor(
                self.mixed_img, cv2.COLOR_BGR2RGB))
        else:
            self.ax_recon.imshow(self.mixed_img, cmap='gray')

        title = f"Reconstructed Image (Mag from Img{self.use_mag_from}: {self.mag_percent}%, Phase from Img{self.use_phase_from}: {self.phase_percent}%)"
        self.ax_recon.set_title(title)
        self.ax_recon.axis('off')
        self.canvas_recon.draw()

    def update_display_mode(self, val=None):
        self.display_mode = self.display_var.get()
        self.process_dft()

    def process_dft(self):
        if self.img1_color is not None:
            self.axs_dft[0, 0].clear()
            self.axs_dft[0, 0].imshow(cv2.cvtColor(
                self.img1_color, cv2.COLOR_BGR2RGB))
            self.axs_dft[0, 0].set_title('Image 1')
            self.axs_dft[0, 0].axis('off')

            dft1 = cv2.dft(np.float32(self.img1_gray),
                           flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift1 = np.fft.fftshift(dft1)
            self.dft_shift1 = dft_shift1
            # Always recompute so a newly uploaded image never shows a stale spectrum
            self.mag1 = cv2.magnitude(
                dft_shift1[:, :, 0], dft_shift1[:, :, 1])
            self.pha1 = np.angle(
                dft_shift1[:, :, 0] + 1j * dft_shift1[:, :, 1])

            self.axs_dft[0, 1].clear()
            if self.display_mode == "magnitude":
                self.axs_dft[0, 1].imshow(np.log(self.mag1 + 1), cmap='gray')
                self.axs_dft[0, 1].set_title('Image 1 Magnitude Spectrum')
            else:
                self.axs_dft[0, 1].imshow(self.pha1, cmap='gray')
                self.axs_dft[0, 1].set_title('Image 1 Phase Spectrum')
            self.axs_dft[0, 1].axis('off')

        if self.img2_color is not None:
            self.axs_dft[1, 0].clear()
            self.axs_dft[1, 0].imshow(cv2.cvtColor(
                self.img2_color, cv2.COLOR_BGR2RGB))
            self.axs_dft[1, 0].set_title('Image 2')
            self.axs_dft[1, 0].axis('off')

            dft2 = cv2.dft(np.float32(self.img2_gray),
                           flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift2 = np.fft.fftshift(dft2)
            self.dft_shift2 = dft_shift2
            # Always recompute so a newly uploaded image never shows a stale spectrum
            self.mag2 = cv2.magnitude(
                dft_shift2[:, :, 0], dft_shift2[:, :, 1])
            self.pha2 = np.angle(
                dft_shift2[:, :, 0] + 1j * dft_shift2[:, :, 1])

            self.axs_dft[1, 1].clear()
            if self.display_mode == "magnitude":
                self.axs_dft[1, 1].imshow(np.log(self.mag2 + 1), cmap='gray')
                self.axs_dft[1, 1].set_title('Image 2 Magnitude Spectrum')
            else:
                self.axs_dft[1, 1].imshow(self.pha2, cmap='gray')
                self.axs_dft[1, 1].set_title('Image 2 Phase Spectrum')
            self.axs_dft[1, 1].axis('off')

        if self.img1_color is not None and self.img2_color is not None:
            self.update_recon_params()

        self.canvas_dft.draw()

    def upload_and_update_dft(self, img_num):
        if self.upload_dft_image(img_num):
            self.process_dft()

    def run(self):
        self.root.mainloop()


def main():
    processor = ImageProcessor()
    processor.run()


if __name__ == "__main__":
    main()

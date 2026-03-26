import cv2
import numpy as np
import logging
from scipy.stats import linregress
from scipy.signal import find_peaks

def perform_spectral_analysis(image_path):
    """
    Layer 3: Spectral Frequency Analysis using Azimuthal Power Spectral Density.
    
    Research basis (Frank et al., ICML 2020; Durall et al., 2020):
    - Natural images follow a 1/f^β power law in the frequency domain (β ≈ 1.5-2.5).
    - GANs fail to replicate this natural decay, showing β < 1.2 or erratic slopes.
    - GAN upsampling (transposed convolution) creates periodic spikes at frequencies 
      corresponding to the stride pattern (checkerboard artifacts).
    - Diffusion models show unnaturally smooth spectral profiles (β > 2.8).
    
    Method:
    1. Compute 2D FFT magnitude spectrum.
    2. Compute azimuthal average (radial power spectral density) — average power in 
       concentric rings from center to edge.
    3. Fit log-log linear regression to get spectral decay slope (beta).
    4. Detect periodic peaks in the radial PSD (GAN signature).
    """
    result = {
        'status': 'error',
        'generator_type': 'Unknown',
        'fft_score': 0.0,         # Legacy compatibility (kept for reasoning engine)
        'fft_beta': 0.0,          # Spectral decay slope
        'fft_peaks': 0,           # Number of periodic spikes detected
        'fft_r_squared': 0.0,     # How well the image fits the 1/f model
        'details': 'Analysis failed.'
    }

    try:
        # Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            result['details'] = 'Failed to load image for spectral analysis.'
            return result

        # Resize to standard size for consistent analysis
        target_size = 512
        img = cv2.resize(img, (target_size, target_size))

        # Perform 2D FFT
        f = np.fft.fft2(img.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Avoid log(0)
        magnitude = np.maximum(magnitude, 1e-10)

        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2

        # ================================================================
        # AZIMUTHAL AVERAGE (Radial Power Spectral Density)
        # Average power in concentric rings from center to edge
        # ================================================================
        max_radius = min(crow, ccol)
        radial_psd = np.zeros(max_radius)
        radial_counts = np.zeros(max_radius)
        
        # Create distance matrix from center
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        distances = np.sqrt(x*x + y*y).astype(int)
        
        # Accumulate power per radius
        for r in range(max_radius):
            ring_mask = (distances == r)
            ring_values = magnitude[ring_mask]
            if len(ring_values) > 0:
                radial_psd[r] = np.mean(ring_values)
                radial_counts[r] = len(ring_values)
        
        # Skip DC component (index 0) and very low frequencies
        start_freq = 5
        valid_psd = radial_psd[start_freq:]
        
        if len(valid_psd) < 20:
            result['details'] = 'Image too small for spectral analysis.'
            return result

        # ================================================================
        # SPECTRAL DECAY SLOPE (Beta) via log-log linear regression
        # Natural images: β ≈ 1.5 to 2.5
        # GANs: β < 1.2 (fails to replicate high-freq decay)
        # Diffusion: β > 2.8 (over-smooth)
        # ================================================================
        frequencies = np.arange(start_freq, start_freq + len(valid_psd)) + 1
        
        # Log-log space
        log_freq = np.log10(frequencies.astype(np.float64))
        log_power = np.log10(valid_psd + 1e-10)
        
        # Linear regression in log-log space: log(P) = -β * log(f) + c
        slope, intercept, r_value, p_value, std_err = linregress(log_freq, log_power)
        beta = -slope  # Negate because power DECREASES with frequency
        r_squared = r_value ** 2
        
        result['fft_beta'] = round(float(beta), 3)
        result['fft_r_squared'] = round(float(r_squared), 3)
        
        # ================================================================
        # PERIODIC PEAK DETECTION (GAN checkerboard signature)
        # GANs create spikes at regular intervals in the radial PSD
        # ================================================================
        # Detrend the PSD to make peaks visible
        expected_psd = 10 ** (slope * log_freq + intercept)
        residual = log_power - np.log10(expected_psd + 1e-10)
        
        # Find peaks in the detrended residual
        # Prominence threshold filters out noise
        peaks, properties = find_peaks(residual, prominence=0.15, distance=5)
        num_peaks = len(peaks)
        result['fft_peaks'] = int(num_peaks)

        # ================================================================
        # LEGACY SCORE (for backward compatibility with reasoning engine)
        # Also compute the old CoV metric
        # ================================================================
        r_quarter = min(rows, cols) // 4
        mask = x*x + y*y >= r_quarter*r_quarter
        high_freq_mag = 20 * np.log(magnitude[mask] + 1e-8)
        mean_hf = np.mean(high_freq_mag)
        std_hf = np.std(high_freq_mag)
        fft_score = float(std_hf / (mean_hf + 1e-8))
        result['fft_score'] = round(fft_score, 4)

        # ================================================================
        # CLASSIFICATION
        # ================================================================
        detail_parts = []
        
        # Primary signal: spectral decay slope
        if beta < 0.8:
            generator = 'GAN / Spectral Decay Failure'
            detail_parts.append(f"Spectral decay β={beta:.2f} is far below natural range (0.9–2.5) — characteristic of GAN upsampling artifacts.")
        elif beta < 1.0:
            generator = 'GAN / Weak Spectral Decay'
            detail_parts.append(f"Spectral decay β={beta:.2f} is below natural range — possible GAN origin with some post-processing.")
        elif beta > 3.0:
            generator = 'Diffusion / Over-Smooth Spectrum'
            detail_parts.append(f"Spectral decay β={beta:.2f} exceeds natural range — unnaturally smooth frequency roll-off typical of diffusion models.")
        elif beta > 2.6:
            generator = 'Diffusion / Slightly Smooth'
            detail_parts.append(f"Spectral decay β={beta:.2f} is at the upper edge of natural range — possible diffusion model origin.")
        else:
            generator = 'Natural / Normal Decay'
            detail_parts.append(f"Spectral decay β={beta:.2f} falls within natural camera physics range (0.9–2.5).")
        
        # Secondary signal: periodic peaks
        if num_peaks >= 5:
            if 'Natural' in generator:
                generator = 'GAN / Periodic Artifacts Detected'
            detail_parts.append(f"Detected {num_peaks} periodic frequency spikes — indicative of GAN checkerboard artifacts.")
        elif num_peaks >= 3:
            detail_parts.append(f"Found {num_peaks} minor frequency anomalies — possible subtle artifacts.")
        
        # Fit quality
        if r_squared < 0.7:
            detail_parts.append(f"Poor 1/f power law fit (R²={r_squared:.2f}) — spectrum does not follow natural image statistics.")
        
        result['generator_type'] = generator
        result['details'] = ' | '.join(detail_parts)
        result['status'] = 'success'

    except Exception as e:
        logging.error(f"Spectral Analysis Error: {str(e)}")
        result['details'] = f'Error during analysis: {str(e)}'

    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        res = perform_spectral_analysis(sys.argv[1])
        for k, v in res.items():
            print(f"  {k}: {v}")

import cv2
import numpy as np
import logging

def perform_spectral_analysis(image_path):
    """
    Analyzes the frequency spectrum of an image using 2D FFT.
    GANs (Generative Adversarial Networks) tend to leave periodic artifacts
    due to upsampling layers, which appear as high-frequency spikes.
    Diffusion Models tend to have smoother frequency roll-offs.
    """
    result = {
        'status': 'error',
        'generator_type': 'Unknown',
        'fft_score': 0.0,
        'details': 'Analysis failed.'
    }

    try:
        # Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            result['details'] = 'Failed to load image for spectral analysis.'
            return result

        # Perform 2D Fast Fourier Transform
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        # We care about the high-frequency details (outer regions of the shifted spectrum)
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create a mask to filter out the low frequencies (center part)
        # r is the radius of the low-frequency area to ignore
        r = min(rows, cols) // 4
        
        # Get coordinates grid
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        mask = x*x + y*y >= r*r
        
        # Apply mask to the magnitude spectrum
        high_freq_mag = magnitude_spectrum[mask]
        
        if len(high_freq_mag) == 0:
            result['details'] = 'Image too small for spectral analysis.'
            return result

        # Analyze the high frequency components
        # A GAN typically has spikes due to checkerboard artifacts.
        # Spikes mean high variance or large max/mean ratio in high frequencies.
        mean_hf = np.mean(high_freq_mag)
        std_hf = np.std(high_freq_mag)
        
        # Calculate a "Spikiness Score" (Coefficient of Variation)
        fft_score = float(std_hf / (mean_hf + 1e-8))
        result['fft_score'] = fft_score

        # Calibrated Heuristic Thresholds
        # GANs exhibit high variance > 0.150
        # Diffusion models exhibit abnormal smoothness < 0.095
        
        if fft_score > 0.150:
            result['generator_type'] = 'GAN / Periodic Spikes'
            result['details'] = f'Detected periodic high-frequency checkerboard artifacts (Score: {fft_score:.2f}).'
            result['status'] = 'success'
        elif fft_score < 0.095:
            result['generator_type'] = 'Diffusion / Unnatural Smoothness'
            result['details'] = f'Detected unnaturally smooth frequency roll-off typical of AI diffusion (Score: {fft_score:.2f}).'
            result['status'] = 'success'
        else:
            result['generator_type'] = 'Inconclusive (Normal Roll-off)'
            result['details'] = f'Frequency spectrum falls within natural camera physics variance (Score: {fft_score:.2f}).'
            result['status'] = 'success'

    except Exception as e:
        logging.error(f"Spectral Analysis Error: {str(e)}")
        result['details'] = f'Error during analysis: {str(e)}'

    return result

if __name__ == '__main__':
    # Simple test if run directly
    import sys
    if len(sys.argv) > 1:
        res = perform_spectral_analysis(sys.argv[1])
        print(res)

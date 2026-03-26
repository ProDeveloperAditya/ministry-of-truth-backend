import cv2
import numpy as np
import logging
from PIL import Image, ImageChops
from scipy.stats import kurtosis
import io

def analyze_ela_and_noise(image_path):
    """
    Layer 4: Multi-Quality Error Level Analysis + Sensor Noise Forensics.
    
    Research basis:
    - Multi-quality ELA (Krawetz, 2007): Performing ELA at multiple JPEG qualities 
      and measuring variance across them. Real images (with compression history) show 
      variable ELA. AI images (no compression history) show uniform ELA.
    - PRNU proxy via noise residual analysis: Real camera noise follows near-Gaussian 
      distribution (kurtosis ≈ 3). AI noise is either absent or has unnatural distribution.
    - Spatial uniformity: Real images have spatially varying noise (due to ISO/exposure). 
      AI images have spatially uniform noise.
    
    Returns dict with:
      - ela_variance: Legacy single-quality ELA variance
      - ela_multi_std: Std dev of ELA scores across multiple qualities (higher = more real)
      - noise_variance: Raw noise residual variance
      - noise_kurtosis: Kurtosis of noise distribution (≈3 for Gaussian/real, extreme for AI)
      - noise_spatial_uniformity: How uniform the noise is across blocks (0-1, higher = more uniform = more AI)
    """
    result = {
        'status': 'error',
        'ela_variance': 0.0,
        'ela_multi_std': 0.0,
        'noise_variance': 0.0,
        'noise_kurtosis': 0.0,
        'noise_spatial_uniformity': 0.0,
        'generator_type': 'Inconclusive Physics',
        'details': 'Analysis failed.'
    }
    
    try:
        # ================================================================
        # 1. MULTI-QUALITY ELA
        # ================================================================
        original = Image.open(image_path).convert('RGB')
        original_np = np.array(original).astype(np.float64)
        
        qualities = [95, 90, 85, 75]
        ela_scores = []
        
        for quality in qualities:
            temp_io = io.BytesIO()
            original.save(temp_io, 'JPEG', quality=quality)
            temp_io.seek(0)
            resaved = Image.open(temp_io)
            resaved_np = np.array(resaved).astype(np.float64)
            
            # Per-pixel absolute difference
            diff = np.abs(original_np - resaved_np)
            ela_score = np.mean(diff)
            ela_scores.append(ela_score)
        
        # Single-quality ELA variance (legacy, at q=90)
        temp_io = io.BytesIO()
        original.save(temp_io, 'JPEG', quality=90)
        temp_io.seek(0)
        resaved_90 = Image.open(temp_io)
        ela_img = ImageChops.difference(original, resaved_90)
        extrema = ela_img.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0: max_diff = 1
        scale = 255.0 / max_diff
        ela_enhanced = Image.eval(ela_img, lambda x: x * scale)
        ela_np = np.array(ela_enhanced)
        ela_variance = float(np.var(ela_np) / 255.0)
        
        # Multi-quality metric: std dev across ELA scores at different qualities
        # Real images: high std (different compression histories at different qualities)
        # AI images: low std (no compression history, uniform response)
        ela_multi_std = float(np.std(ela_scores))
        
        result['ela_variance'] = round(ela_variance, 4)
        result['ela_multi_std'] = round(ela_multi_std, 4)
        
        # ================================================================
        # 2. SENSOR NOISE ANALYSIS
        # ================================================================
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            result['details'] = 'Failed to load image for noise analysis.'
            return result
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # Better noise extraction: Gaussian blur (preserves edges better than median)
        # then subtract to isolate noise
        denoised = cv2.GaussianBlur(gray, (5, 5), 2.0)
        noise_residual = gray - denoised
        
        noise_variance = float(np.var(noise_residual))
        result['noise_variance'] = round(noise_variance, 4)
        
        # Noise kurtosis: measures the "tailedness" of the noise distribution
        # Real camera noise (shot noise + PRNU): near-Gaussian → kurtosis ≈ 3.0
        # AI "noise" (if any): highly non-Gaussian → kurtosis >> 3 or << 3
        # No noise at all: kurtosis → 0 or undefined
        noise_flat = noise_residual.flatten()
        if np.std(noise_flat) > 0.01:
            noise_kurt = float(kurtosis(noise_flat, fisher=False))  # fisher=False → excess kurtosis + 3
        else:
            noise_kurt = 0.0  # Essentially zero noise
        result['noise_kurtosis'] = round(noise_kurt, 3)
        
        # ================================================================
        # 3. SPATIAL NOISE UNIFORMITY (Block Analysis)
        # ================================================================
        # Divide image into 8x8 grid, compute noise variance per block
        # Real images: variance varies across blocks (due to scene content, ISO)
        # AI images: variance is suspiciously uniform
        h, w = gray.shape
        block_h, block_w = h // 8, w // 8
        
        if block_h > 8 and block_w > 8:
            block_variances = []
            for i in range(8):
                for j in range(8):
                    block = noise_residual[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                    block_variances.append(np.var(block))
            
            block_variances = np.array(block_variances)
            mean_bv = np.mean(block_variances)
            
            if mean_bv > 0.01:
                # Coefficient of variation of block variances
                # Low CoV = uniform noise = suspicious (AI)
                # High CoV = variable noise = natural (camera)
                cov_blocks = float(np.std(block_variances) / mean_bv)
                # Invert and normalize: 0 = highly variable (real), 1 = uniform (AI)
                spatial_uniformity = max(0.0, min(1.0, 1.0 - cov_blocks))
            else:
                spatial_uniformity = 1.0  # Zero noise everywhere = very uniform = AI
        else:
            spatial_uniformity = 0.5  # Image too small
        
        result['noise_spatial_uniformity'] = round(spatial_uniformity, 3)
        
        # ================================================================
        # CLASSIFICATION
        # ================================================================
        result['status'] = 'success'
        detail_parts = []
        gen_type = 'Inconclusive Physics'
        
        # Signal 1: Noise variance
        if noise_variance < 1.0:
            gen_type = 'AI / Zero Sensor Noise'
            detail_parts.append(f"Near-zero noise variance ({noise_variance:.2f}) — no camera sensor noise present.")
        elif noise_variance < 5.0:
            detail_parts.append(f"Very low noise variance ({noise_variance:.2f}) — minimal sensor noise.")
        
        # Signal 2: Noise kurtosis
        if noise_kurt == 0.0:
            detail_parts.append(f"No measurable noise distribution — image is mathematically smooth.")
            gen_type = 'AI / Mathematically Smooth'
        
        # Signal 3: Spatial uniformity
        if spatial_uniformity > 0.80:
            detail_parts.append(f"Spatially uniform noise ({spatial_uniformity:.2f}) — real cameras produce varying noise across the frame.")
            if gen_type == 'Inconclusive Physics':
                gen_type = 'Suspicious / Uniform Noise'
        
        # Signal 4: Multi-quality ELA
        if ela_multi_std < 0.5:
            detail_parts.append(f"Uniform ELA response across compression qualities (std: {ela_multi_std:.2f}) — no JPEG compression history detected.")
            if gen_type == 'Inconclusive Physics':
                gen_type = 'Suspicious / No Compression History'
        elif ela_multi_std > 3.0:
            detail_parts.append(f"Strong ELA variance across qualities (std: {ela_multi_std:.2f}) — image has natural compression history.")
        
        # Signal 5: ELA variance (legacy)
        if ela_variance > 1.25 and noise_variance < 80.0:
            detail_parts.append(f"Inconsistent compression mapping (ELA: {ela_variance:.2f}).")
            if gen_type == 'Inconclusive Physics':
                gen_type = 'AI / ELA Anomaly'
        
        if not detail_parts:
            detail_parts.append(f'Physics values nominal (ELA: {ela_variance:.2f}, Noise: {noise_variance:.2f}, Kurtosis: {noise_kurt:.1f}).')
        
        result['generator_type'] = gen_type
        result['details'] = ' | '.join(detail_parts)
        
    except Exception as e:
        logging.error(f"ELA/Noise Error: {str(e)}")
        result['details'] = f'Error during analysis: {str(e)}'
        
    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        res = analyze_ela_and_noise(sys.argv[1])
        for k, v in res.items():
            print(f"  {k}: {v}")

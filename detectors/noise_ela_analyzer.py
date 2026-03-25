import cv2
import numpy as np
import logging
from PIL import Image, ImageChops
import io

def analyze_ela_and_noise(image_path):
    """
    Performs Error Level Analysis (ELA) and Sensor Noise Extration.
    - ELA measures differential JPEG compression rates (AI often has mismatched rates).
    - Sensor Noise (PRNU proxy) measures the natural chaos of a hardware camera vs mathematical smoothness.
    """
    result = {
        'status': 'error',
        'ela_variance': 0.0,
        'noise_variance': 0.0,
        'details': 'Analysis failed.'
    }
    
    try:
        # 1. ELA (Error Level Analysis)
        original = Image.open(image_path).convert('RGB')
        
        # Save to memory at known quality
        temp_io = io.BytesIO()
        original.save(temp_io, 'JPEG', quality=90)
        temp_io.seek(0)
        resaved = Image.open(temp_io)
        
        # Calculate pixel difference
        ela_img = ImageChops.difference(original, resaved)
        extrema = ela_img.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        # Enhance for variance calc
        if max_diff == 0: max_diff = 1 # avoid div by zero
        scale = 255.0 / max_diff
        ela_enhanced = Image.eval(ela_img, lambda x: x * scale)
        
        # Convert to numpy for stats
        ela_np = np.array(ela_enhanced)
        ela_variance = np.var(ela_np) / 255.0  # Normalize
        
        # 2. Sensor Noise Variance
        img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Median filter removes noise but keeps structure
        blurred = cv2.medianBlur(img_cv, 3)
        # Difference leaves ONLY the noise
        noise = cv2.absdiff(img_cv, blurred)
        
        noise_variance = np.var(noise)
        
        result['status'] = 'success'
        result['ela_variance'] = float(ela_variance)
        result['noise_variance'] = float(noise_variance)
        
        # Analyze and provide component level verdicts
        text_details = []
        gen_type = 'Inconclusive Physics'
        
        if ela_variance > 1.25 and noise_variance < 80.0:
            gen_type = 'AI / ELA Anomaly'
            text_details.append(f"Inconsistent AI compression mapping (ELA: {ela_variance:.2f})")
            
        if noise_variance < 5.0:
            gen_type = 'AI / Unnaturally Smooth'
            text_details.append(f"Zero camera sensor noise (Noise: {noise_variance:.2f})")
            
        if not text_details:
            text_details.append(f'Physics values nominal (ELA Var: {ela_variance:.2f}, Noise Var: {noise_variance:.2f}).')
            
        result['generator_type'] = gen_type
        result['details'] = " | ".join(text_details)
        
    except Exception as e:
        logging.error(f"ELA/Noise Error: {str(e)}")
        
    return result

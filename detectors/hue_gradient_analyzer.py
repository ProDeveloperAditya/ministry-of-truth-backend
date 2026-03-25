import cv2
import numpy as np
import base64
import logging

def analyze_hue_gradient(image_path, return_mask=True):
    """
    Analyzes the Hue and Value channels of an image to detect unnatural 
    gradient transitions ("stitching gaps") characteristic of AI generators.
    """
    result = {
        'status': 'error',
        'edge_density_score': 0.0,
        'generator_type': 'Unknown',
        'details': 'Analysis failed.',
        'visual_mask_base64': None
    }
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            result['details'] = 'Failed to load image for Hue analysis.'
            return result
        
        # Resize to standard width to normalize edge detection thresholds
        # Keep aspect ratio
        target_width = 800
        h, w = img.shape[:2]
        if w > target_width:
            scale = target_width / float(w)
            img = cv2.resize(img, (target_width, int(h * scale)))
            
        # --- RESEARCH IMPLEMENTATION: YCbCr Chrominance Disconnect ---
        # AI models (GANs/Diffusion) generate RGB arrays from Latent Space, bypassing 
        # the physical Bayer filter demosaicing of real camera sensors. This creates severe
        # misalignments between physical structure (Luminance) and color (Chrominance).
        
        # Convert to YCrCb (Y=Luminance/Structure, Cr/Cb=Chrominance/Color)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Calculate highly precise gradients using Scharr operator
        grad_y_x = cv2.Scharr(y, cv2.CV_64F, 1, 0)
        grad_y_y = cv2.Scharr(y, cv2.CV_64F, 0, 1)
        grad_y_mag = cv2.magnitude(grad_y_x, grad_y_y)
        
        grad_cr_x = cv2.Scharr(cr, cv2.CV_64F, 1, 0)
        grad_cr_y = cv2.Scharr(cr, cv2.CV_64F, 0, 1)
        grad_cr_mag = cv2.magnitude(grad_cr_x, grad_cr_y)
        
        grad_cb_x = cv2.Scharr(cb, cv2.CV_64F, 1, 0)
        grad_cb_y = cv2.Scharr(cb, cv2.CV_64F, 0, 1)
        grad_cb_mag = cv2.magnitude(grad_cb_x, grad_cb_y)
        
        # Combine Chrominance gradients into one color-shift matrix
        chroma_mag = grad_cr_mag + grad_cb_mag
        
        # Normalize and convert to 8-bit for thresholding
        cv2.normalize(grad_y_mag, grad_y_mag, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(chroma_mag, chroma_mag, 0, 255, cv2.NORM_MINMAX)
        grad_y_8u = np.uint8(grad_y_mag)
        chroma_8u = np.uint8(chroma_mag)
        
        # Otsu's Thresholding to dynamically find the strongest physical and color edges
        _, y_edges = cv2.threshold(grad_y_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, chroma_edges = cv2.threshold(chroma_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Real camera lenses have slight chromatic aberration, so color edges 
        # might bleed 1-2 pixels out of the physical structure. Dilate Y to compensate.
        kernel = np.ones((5,5), np.uint8)
        y_edges_dilated = cv2.dilate(y_edges, kernel, iterations=1)
        
        # The AI Anomaly: Find pixels where the AI created a sharp color edge, 
        # but there is literally zero physical sub-structure behind it.
        anomaly_mask = cv2.bitwise_and(chroma_edges, cv2.bitwise_not(y_edges_dilated))
        
        # Calculate the Disconnect Score
        total_chroma_pixels = np.count_nonzero(chroma_edges)
        anomaly_pixels = np.count_nonzero(anomaly_mask)
        
        if total_chroma_pixels == 0: total_chroma_pixels = 1
        disconnect_score = anomaly_pixels / float(total_chroma_pixels)
        
        result['edge_density_score'] = disconnect_score
        
        # Based on research, real images rarely exceed 0.25 color-structural disconnect.
        threshold = 0.35 
        
        if disconnect_score > threshold:
            result['status'] = 'success'
            result['generator_type'] = 'AI / Chrominance Stitching Detected'
            result['details'] = f'Found impossible Chrominance edges disconnected from Luminance shapes (Score: {disconnect_score:.2f}).'
        else:
            result['status'] = 'success'
            result['generator_type'] = 'Inconclusive (Correlated)'
            result['details'] = f'Color gradients cleanly adhere to structural shape edges (Score: {disconnect_score:.2f}).'
            
        if return_mask:
            # We highlight the exact pixels where the AI failed to align Latent variables
            gray_base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Dim the background to make red pop
            gray_base = (gray_base * 0.4).astype(np.uint8)
            canvas = cv2.cvtColor(gray_base, cv2.COLOR_GRAY2BGR)
            
            # Make the anomalous edges thicker so the user can easily see them
            display_mask = cv2.dilate(anomaly_mask, np.ones((3,3), np.uint8), iterations=1)
            canvas[display_mask > 0] = [0, 0, 255] # Pure Red
            
            _, buffer = cv2.imencode('.jpg', canvas)
            result['visual_mask_base64'] = base64.b64encode(buffer).decode('utf-8')

        
    except Exception as e:
        logging.error(f"Hue Gradient Analysis Error: {str(e)}")
        result['details'] = f'Error during analysis: {str(e)}'
        
    return result

if __name__ == '__main__':
    # Test script
    import sys
    if len(sys.argv) > 1:
        res = analyze_hue_gradient(sys.argv[1])
        print({k: v for k,v in res.items() if k != 'visual_mask_base64'})

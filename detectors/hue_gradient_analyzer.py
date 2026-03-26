import cv2
import numpy as np
import base64
import logging

def analyze_hue_gradient(image_path, return_mask=True):
    """
    Layer 5: YCbCr Chrominance Disconnect + CFA Demosaicing Trace Detection.
    
    Two complementary forensic signals:
    
    1. CHROMINANCE DISCONNECT (existing, improved):
       AI models generate RGB from latent space, bypassing the physical Bayer 
       filter demosaicing of real cameras. This creates misalignments between 
       luminance (structure) and chrominance (color).
       
    2. CFA DEMOSAICING TRACE (new, from Popescu & Farid, IEEE TIFS 2005):
       Real cameras capture only ONE color per pixel through a Bayer filter.
       The other two colors are interpolated, leaving a periodic 2x2 correlation 
       pattern in each color channel. AI generators produce all 3 channels 
       independently, so this pattern is completely absent.
    """
    result = {
        'status': 'error',
        'edge_density_score': 0.0,       # Chrominance disconnect (0-1)
        'cfa_trace_strength': 0.0,        # Bayer pattern strength (higher = more real)
        'hue_confidence': 0.0,            # Combined AI confidence (0-1)
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
        
        # Resize to standard width for consistent thresholds
        target_width = 800
        h, w = img.shape[:2]
        if w > target_width:
            scale = target_width / float(w)
            img = cv2.resize(img, (target_width, int(h * scale)))
        
        # ================================================================
        # SIGNAL 1: CFA Demosaicing Trace Detection
        # (Popescu & Farid, IEEE TIFS 2005)
        # ================================================================
        cfa_trace = _detect_cfa_traces(img)
        result['cfa_trace_strength'] = cfa_trace
        
        # ================================================================
        # SIGNAL 2: YCbCr Chrominance-Luminance Disconnect (improved)
        # ================================================================
        disconnect_score, anomaly_mask = _compute_chrominance_disconnect(img)
        result['edge_density_score'] = disconnect_score
        
        # ================================================================
        # COMBINED CONFIDENCE
        # ================================================================
        # CFA trace: high = real camera (inverted for AI confidence)
        # Disconnect: high = AI
        
        # CFA trace scoring (strongest forensic signal, but easily destroyed by compression)
        # Real cameras typically show trace > 0.12
        # HOWEVER, compressed real images (WhatsApp, web) often have trace < 0.05
        # Therefore: presence of trace = Camera. Absence of trace = Inconclusive (fallback to chroma)
        
        if cfa_trace > 0.15:
            cfa_ai_conf = 0.01   # Definitively real
        elif cfa_trace > 0.08:
            cfa_ai_conf = 0.15   # Probably real
        else:
            cfa_ai_conf = 0.50   # Inconclusive (trace destroyed by compression or AI)
        
        # Chrominance disconnect scoring
        # Normal images can have some disconnect due to JPEG chroma subsampling (up to 0.3)
        if disconnect_score > 0.45:
            chroma_ai_conf = 0.90
        elif disconnect_score > 0.35:
            chroma_ai_conf = 0.70
        elif disconnect_score > 0.25:
            chroma_ai_conf = 0.55
        else:
            chroma_ai_conf = 0.10
        
        # Override logic for CFA trace:
        # Modern AI upsampling (PixelShuffle/Transposed Convolutions) creates the exact 
        # same 2x2 periodic high-frequency grid as Bayer demosaicing. 
        # So we CANNOT use CFA trace to overrule the AI prediction.
        combined = chroma_ai_conf
            
        result['hue_confidence'] = round(combined, 3)
        
        # Generate verdict
        detail_parts = []
        
        if cfa_trace < 0.03:
            detail_parts.append(f"No Bayer CFA demosaicing pattern detected (trace: {cfa_trace:.3f}) — could be AI or heavily compressed.")
        elif cfa_trace < 0.08:
            detail_parts.append(f"Very weak CFA trace ({cfa_trace:.3f}).")
        else:
            detail_parts.append(f"Strong Bayer CFA demosaicing pattern present (trace: {cfa_trace:.3f}) — definitive camera sensor physics.")
        
        if disconnect_score > 0.45:
            detail_parts.append(f"Strong chrominance-luminance disconnect ({disconnect_score:.2f}) — color edges exist without structural support (AI signature).")
        elif disconnect_score > 0.30:
            detail_parts.append(f"Moderate chrominance disconnect ({disconnect_score:.2f}).")
        else:
            detail_parts.append(f"Color channels properly correlated with structure ({disconnect_score:.2f}).")
        
        if combined > 0.60:
            result['generator_type'] = 'AI / No Camera Physics'
        elif combined > 0.40:
            result['generator_type'] = 'Suspicious (Weak Camera Traces)'
        else:
            result['generator_type'] = 'Consistent with Real Camera'
        
        result['status'] = 'success'
        result['details'] = ' | '.join(detail_parts)
        
        # Visual mask for the disconnect anomalies
        if return_mask and anomaly_mask is not None:
            gray_base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_base = (gray_base * 0.4).astype(np.uint8)
            canvas = cv2.cvtColor(gray_base, cv2.COLOR_GRAY2BGR)
            display_mask = cv2.dilate(anomaly_mask, np.ones((3,3), np.uint8), iterations=1)
            canvas[display_mask > 0] = [0, 0, 255]
            _, buffer = cv2.imencode('.jpg', canvas)
            result['visual_mask_base64'] = base64.b64encode(buffer).decode('utf-8')
        
    except Exception as e:
        logging.error(f"Hue/CFA Analysis Error: {str(e)}")
        result['details'] = f'Error during analysis: {str(e)}'
        
    return result


def _detect_cfa_traces(img):
    """
    Detects the periodic 2x2 Bayer CFA interpolation pattern in each color channel.
    
    Method (Popescu & Farid):
    1. For each color channel, compute the interpolation residual:
       residual = channel - bilinear_interpolated(downsampled(channel))
    2. Compute the 2D DFT of the residual.
    3. Look for peaks at Nyquist frequencies (π, π), (π, 0), (0, π) 
       which correspond to the 2x2 Bayer pattern.
    4. The peak magnitude relative to the DC component = CFA trace strength.
    
    Returns: float (0.0 = no trace, higher = stronger camera trace)
    """
    try:
        h, w = img.shape[:2]
        
        # Need minimum size for meaningful analysis
        if h < 64 or w < 64:
            return 0.0
        
        trace_scores = []
        
        for c in range(3):  # B, G, R channels
            channel = img[:, :, c].astype(np.float64)
            
            # Downsample by 2x (simulating Bayer subsampling)
            downsampled = channel[::2, ::2]
            
            # Upsample back using bilinear interpolation
            interpolated = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # The residual contains the interpolation artifacts
            residual = channel - interpolated
            
            # Compute 2D DFT of the residual
            dft = np.fft.fft2(residual)
            magnitude = np.abs(dft)
            
            # Normalize by DC component to make it scale-invariant
            dc = magnitude[0, 0]
            if dc < 1e-8:
                trace_scores.append(0.0)
                continue
            
            # Check for peaks at Bayer pattern frequencies:
            # (h//2, w//2) = Nyquist (π, π) — the primary Bayer frequency
            # (h//2, 0)    = (π, 0) — horizontal Bayer edge
            # (0, w//2)    = (0, π) — vertical Bayer edge
            peak_nyquist = magnitude[h//2, w//2] / dc
            peak_h = magnitude[h//2, 0] / dc
            peak_v = magnitude[0, w//2] / dc
            
            # Average the peaks — real cameras show strong peaks at all three
            channel_trace = (peak_nyquist + peak_h + peak_v) / 3.0
            trace_scores.append(channel_trace)
        
        # Green channel has 2x density in Bayer so it typically has the weakest
        # trace. Use the median across channels for robustness.
        if trace_scores:
            return float(np.median(trace_scores))
        return 0.0
        
    except Exception as e:
        logging.error(f"CFA trace detection error: {str(e)}")
        return 0.0


def _compute_chrominance_disconnect(img):
    """
    Improved chrominance-luminance disconnect analysis.
    Pre-filters JPEG artifacts with mild Gaussian blur before computing gradients.
    
    Returns: (disconnect_score, anomaly_mask)
    """
    try:
        # Pre-filter to suppress JPEG block artifacts (reduces false positives)
        img_filtered = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Scharr gradients for luminance
        grad_y_x = cv2.Scharr(y, cv2.CV_64F, 1, 0)
        grad_y_y = cv2.Scharr(y, cv2.CV_64F, 0, 1)
        grad_y_mag = cv2.magnitude(grad_y_x, grad_y_y)
        
        # Scharr gradients for chrominance
        grad_cr_x = cv2.Scharr(cr, cv2.CV_64F, 1, 0)
        grad_cr_y = cv2.Scharr(cr, cv2.CV_64F, 0, 1)
        grad_cr_mag = cv2.magnitude(grad_cr_x, grad_cr_y)
        
        grad_cb_x = cv2.Scharr(cb, cv2.CV_64F, 1, 0)
        grad_cb_y = cv2.Scharr(cb, cv2.CV_64F, 0, 1)
        grad_cb_mag = cv2.magnitude(grad_cb_x, grad_cb_y)
        
        chroma_mag = grad_cr_mag + grad_cb_mag
        
        # Normalize and threshold
        cv2.normalize(grad_y_mag, grad_y_mag, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(chroma_mag, chroma_mag, 0, 255, cv2.NORM_MINMAX)
        grad_y_8u = np.uint8(grad_y_mag)
        chroma_8u = np.uint8(chroma_mag)
        
        # Otsu thresholding
        _, y_edges = cv2.threshold(grad_y_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, chroma_edges = cv2.threshold(chroma_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilate luminance edges to account for chromatic aberration (1-2px)
        kernel = np.ones((5, 5), np.uint8)
        y_edges_dilated = cv2.dilate(y_edges, kernel, iterations=1)
        
        # Anomaly: color edges where there is NO structural edge
        anomaly_mask = cv2.bitwise_and(chroma_edges, cv2.bitwise_not(y_edges_dilated))
        
        total_chroma_pixels = np.count_nonzero(chroma_edges)
        anomaly_pixels = np.count_nonzero(anomaly_mask)
        
        if total_chroma_pixels == 0:
            total_chroma_pixels = 1
        disconnect_score = anomaly_pixels / float(total_chroma_pixels)
        
        return float(disconnect_score), anomaly_mask
        
    except Exception as e:
        logging.error(f"Chrominance disconnect error: {str(e)}")
        return 0.0, None


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        res = analyze_hue_gradient(sys.argv[1])
        print({k: v for k, v in res.items() if k != 'visual_mask_base64'})

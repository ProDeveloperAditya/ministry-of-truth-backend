import os
import sys
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import json

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.hue_gradient_analyzer import analyze_hue_gradient
from detectors.metadata_analyzer import analyze_metadata_and_structure
from detectors.spectral_analyzer import perform_spectral_analysis
from detectors.noise_ela_analyzer import analyze_ela_and_noise

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/test_analyze', methods=['POST'])
def test_analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Run the 3 new outstanding detectors
        hue_result = analyze_hue_gradient(filepath)
        metadata_result = analyze_metadata_and_structure(filepath)
        spectral_result = perform_spectral_analysis(filepath)
        noise_result = analyze_ela_and_noise(filepath)
        
        # Determine actual metrics
        hue_score = hue_result.get('edge_density_score', 0)
        fft_var = spectral_result.get('fft_score', 0)
        meta_conf = metadata_result.get('ai_confidence', 0)
        ela_var = noise_result.get('ela_variance', 0)
        noise_var = noise_result.get('noise_variance', 100) # Default to high noise if failed
        
        # --- 1. Meta ---
        details = []
        layer_conf_meta = meta_conf
        if meta_conf >= 0.5:
            details.append("Metadata signature (Software Tags / Fixed Dimensions)")
        elif meta_conf == 0.0:
            details.append("Authentic camera hardware EXIF detected")
            
        # --- 2. Chrominance Mismatch (formerly Hue Stitching) ---
        layer_conf_hue = 0.1
        if hue_score > 0.35:
            layer_conf_hue = 0.85
            details.append(f"Severe Chrominance/Structural Edge Disconnect (Mismatch: {hue_score:.2f})")
            
        # --- 3. FFT High-Freq ---
        layer_conf_fft = 0.1
        if fft_var > 0.150:
            layer_conf_fft = 0.75
            details.append(f"High-frequency artifact spikes (FFT: {fft_var:.2f})")
        elif fft_var < 0.095:
            layer_conf_fft = 0.85
            details.append(f"Unnaturally smooth frequency roll-off (Diffusion AI: {fft_var:.2f})")
            
        # --- 4. Physics / ELA & Noise ---
        layer_conf_physics = 0.1
        if noise_var < 5.0:
            layer_conf_physics = 0.90
            details.append(f"Unnaturally smooth / zero camera noise (Noise: {noise_var:.2f})")
        elif ela_var > 1.25 and noise_var < 80.0:
            layer_conf_physics = 0.80
            details.append(f"Inconsistent composite compressions (ELA: {ela_var:.2f})")
        elif fft_var < 0.115 and ela_var > 1.15:
            layer_conf_physics = 0.85
            details.append(f"Physics Contradiction: Smooth frequencies but high ELA variance.")
            
        # Inject individual layer confidence scores to be parsed by JavaScript
        metadata_result['layer_ai_score'] = layer_conf_meta
        hue_result['layer_ai_score'] = layer_conf_hue
        spectral_result['layer_ai_score'] = layer_conf_fft
        noise_result['layer_ai_score'] = layer_conf_physics
        
        # Max Evidentiary Logic: If ANY layer proves it's AI, the overall image is highly suspicious.
        ai_score = max(layer_conf_meta, layer_conf_hue, layer_conf_fft, layer_conf_physics)
        
        # Boost score slightly if multiple anomalies found (to a max of 0.99)
        anomalies_count = sum(1 for conf in [layer_conf_hue, layer_conf_fft, layer_conf_physics] if conf >= 0.7)
        if anomalies_count > 1:
            ai_score = min(0.99, ai_score + (anomalies_count * 0.05))
        
        verdict = "Real / Unknown"
        if ai_score >= 0.7:
            verdict = "Highly Likely AI Generated"
        elif ai_score >= 0.4:
            verdict = "Potentially AI Generated"
            
        result = {
            "status": "success",
            "verdict": verdict,
            "overall_ai_score": round(ai_score, 2),
            "summary_details": details,
            "layers": {
                "hue_analysis": hue_result,
                "metadata_analysis": metadata_result,
                "spectral_analysis": spectral_result,
                "noise_ela_analysis": noise_result
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Clean up
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

if __name__ == '__main__':
    print("Starting Advanced Vision Test Server on port 5050...")
    print("Go to http://localhost:5050 in your browser to test images.")
    app.run(host='0.0.0.0', debug=True, port=5050)

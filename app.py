import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
# Enable CORS just in case
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def serve_frontend():
    return app.send_static_file('index.html')

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Backend is running"})

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    # Save the file temporarily
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        from detectors.c2pa_detector import check_c2pa
        from detectors.sightengine_detector import check_external_api
        from detectors.spectral_analyzer import perform_spectral_analysis
        from reasoning_engine import generate_reasoning
        
        # 1. c2patool check
        c2pa_result = check_c2pa(filepath)
        
        # 2. Sightengine API check
        sightengine_result = check_external_api(filepath)
        
        # 3. Spectral FFT Analyzer check (GAN vs Diffusion)
        spectral_result = perform_spectral_analysis(filepath)
        
        # 4. Generate Reasoning
        final_assessment = generate_reasoning(c2pa_result, sightengine_result, spectral_result)
        
        result = {
            "status": "success",
            "verdict": final_assessment["verdict"],
            "confidence": final_assessment["confidence"],
            "generator": final_assessment["generator"],
            "summary": final_assessment["summary"],
            "detailed_reasons": final_assessment["detailed_reasons"],
            "layers": {
                "c2pa": c2pa_result,
                "sightengine": sightengine_result,
                "spectral": spectral_result
            }
        }
        
        return jsonify(result)
        
    finally:
        # Clean up
        if os.path.exists(filepath):
            pass # Keep it for debugging initially, we'll remove it later
            # os.remove(filepath)

if __name__ == '__main__':
    print("Starting AI Content Detection Backend on port 5001...")
    app.run(host='0.0.0.0', debug=True, use_reloader=False, port=5001)

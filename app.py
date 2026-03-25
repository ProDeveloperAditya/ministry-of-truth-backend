import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import cv2

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def home():
    return jsonify({"status": "active", "service": "AI Detector API"})

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Backend is running"})

# Load detectors and reasoning engine at startup to avoid cold-start delays on first request
try:
    from detectors.c2pa_detector import check_c2pa
    from detectors.sightengine_detector import check_external_api
    from detectors.spectral_analyzer import perform_spectral_analysis
    from detectors.noise_ela_analyzer import analyze_ela_and_noise
    from detectors.hue_gradient_analyzer import analyze_hue_gradient
    from reasoning_engine import generate_reasoning
    print("All 5 detection modules loaded successfully.")
except Exception as e:
    print(f"Error loading detection modules: {e}")

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    print(f"Received analysis request: {request.remote_addr}")
    if 'file' not in request.files:
        print("Error: No file part in request")
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        print("Error: Empty filename")
        return jsonify({"error": "No selected file"}), 400
        
    print(f"Processing file: {file.filename} (Size: {request.content_length} bytes)")
        
    # Save the original file temporarily
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Check if video and extract frame
    is_video = file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
    analysis_filepath = filepath
    
    if is_video:
        try:
            print(f"Video detected: {file.filename}. Extracting middle frame...")
            cap = cv2.VideoCapture(filepath)
            
            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Set to middle frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
            
            success, frame = cap.read()
            if success:
                # Save extracted frame as image
                frame_filename = f"frame_{file.filename.rsplit('.', 1)[0]}.jpg"
                analysis_filepath = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
                cv2.imwrite(analysis_filepath, frame)
                print(f"Extracted frame to {analysis_filepath}")
            else:
                cap.release()
                return jsonify({"error": "Failed to extract frame from video"}), 500
                
            cap.release()
        except Exception as e:
            return jsonify({"error": f"Video processing error: {str(e)}"}), 500
            
    try:
        # 1. c2patool check
        c2pa_result = check_c2pa(analysis_filepath)
        
        # 2. Sightengine API check
        sightengine_result = check_external_api(analysis_filepath)
        
        # 3. Spectral FFT Analyzer check (GAN vs Diffusion)
        spectral_result = perform_spectral_analysis(analysis_filepath)
        
        # 4. Physics / ELA + Noise Analysis
        physics_result = analyze_ela_and_noise(analysis_filepath)
        
        # 5. YCbCr Chrominance Disconnect (Hue)
        hue_result = analyze_hue_gradient(analysis_filepath, return_mask=False)
        
        # 6. Generate Reasoning (5-layer input)
        final_assessment = generate_reasoning(
            c2pa_result, sightengine_result, spectral_result,
            physics_result, hue_result
        )
        
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
                "spectral": spectral_result,
                "physics": physics_result,
                "hue": hue_result
            }
        }
        
        return jsonify(result)
        
    finally:
        # Clean up both original and extracted files
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        if is_video and analysis_filepath != filepath and os.path.exists(analysis_filepath):
            try:
                os.remove(analysis_filepath)
            except:
                pass

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting AI Content Detection Backend on port {port}...")
    app.run(host='0.0.0.0', debug=True, use_reloader=False, port=port)

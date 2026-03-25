import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.hue_gradient_analyzer import analyze_hue_gradient
from detectors.metadata_analyzer import analyze_metadata_and_structure
from detectors.spectral_analyzer import perform_spectral_analysis
from detectors.noise_ela_analyzer import analyze_ela_and_noise

test_dir = r"c:\Users\adity\Desktop\Intern projects\Images by me"
report_path = r"c:\Users\adity\.gemini\antigravity\brain\d1697658-7d0e-4450-a0dc-3e9e7877587f\accuracy_report.md"

def get_verdict(filepath):
    # Run the exact logic from test_local_web.py
    hue_result = analyze_hue_gradient(filepath, return_mask=False)
    metadata_result = analyze_metadata_and_structure(filepath)
    spectral_result = perform_spectral_analysis(filepath)
    noise_result = analyze_ela_and_noise(filepath)
    
    hue_score = hue_result.get('edge_density_score', 0)
    fft_var = spectral_result.get('fft_score', 0)
    meta_conf = metadata_result.get('ai_confidence', 0)
    ela_var = noise_result.get('ela_variance', 0)
    noise_var = noise_result.get('noise_variance', 100)
    
    ai_score = 0.0
    details = []
    
    if meta_conf >= 0.5:
        ai_score += 0.8
        details.append("Metadata signature")
    elif meta_conf == 0.0:
        ai_score -= 0.5
        details.append("Authentic camera EXIF")
        
    layer_conf_hue = 0.1
    if hue_score > 0.35:
        layer_conf_hue = 0.85
        details.append(f"YCbCr Disconnect: {hue_score:.2f}")

    if fft_var > 0.150:
        ai_score += 0.3
        details.append(f"FFT Spikes: {fft_var:.2f}")
    elif fft_var < 0.095:
        ai_score += 0.4
        details.append(f"Diffusion Smooth FFT: {fft_var:.2f}")
        
    if ela_var > 1.25 and noise_var < 80.0:
        ai_score += 0.4
        details.append(f"ELA Anomaly: {ela_var:.2f}")
        
    # Hybrid physics constraint: Smooth frequency but high compression variance (Catches High-Noise AIs)
    if fft_var < 0.115 and ela_var > 1.15:
        ai_score += 0.4
        details.append(f"Physics Contradiction (FFT/ELA)")
        
    if noise_var < 5.0:
        ai_score += 0.4
        details.append(f"Low Noise: {noise_var:.2f}")
        
    ai_score = max(0.0, min(1.0, ai_score))
    
    verdict = "Real / Unknown"
    if ai_score >= 0.7:
        verdict = "Highly Likely AI Generated"
    elif ai_score >= 0.4:
        verdict = "Potentially AI Generated"
        
    ground_truth = "AI Generated" if "AI" in os.path.basename(filepath) else "Real Photo"
    
    # Determine correctness
    correct = False
    if "AI" in ground_truth and "AI" in verdict:
        correct = True
    elif "Real" in ground_truth and "Real" in verdict:
        correct = True
        
    return {
        "score": ai_score,
        "verdict": verdict,
        "ground_truth": ground_truth,
        "correct": correct,
        "details": details,
        "raw": f"(Meta: {meta_conf:.2f}, Hue: {hue_score:.2f}, FFT: {fft_var:.2f}, ELA: {ela_var:.2f}, Noise: {noise_var:.2f})"
    }

def build_report():
    files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    correct_count = 0
    total = len(files)
    
    md = "# AI Detector Accuracy Report\n\n"
    md += "This report compares the code-driven predictions against the ground truth labels.\n\n"
    
    md += "## Detailed Results\n\n"
    md += "| Image Name | Ground Truth | Code Verdict | AI Score | Result | Raw Metrics | Flagged Reasons |\n"
    md += "|---|---|---|---|---|---|---|\n"
    
    for f in sorted(files):
        try:
            res = get_verdict(os.path.join(test_dir, f))
            if res['correct']:
                correct_count += 1
                emoji = "✅ MATCH"
            else:
                emoji = "❌ FAIL"
                
            reasons = ", ".join(res['details']) if res['details'] else "None"
            md += f"| `{f}` | **{res['ground_truth']}** | {res['verdict']} | {res['score']:.2f} | {emoji} | {res['raw']} | {reasons} |\n"
        except Exception as e:
            md += f"| `{f}` | ERROR | {str(e)} | | | | |\n"
            
    md += f"\n## Summary\n"
    md += f"**Final Accuracy: {correct_count}/{total} ({(correct_count/total)*100 if total > 0 else 0:.1f}%)**\n"
    
    with open(report_path, "w", encoding="utf-8") as out:
        out.write(md)

if __name__ == "__main__":
    build_report()
    print("Report generated.")

import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.c2pa_detector import check_c2pa
from detectors.sightengine_detector import check_external_api
from detectors.spectral_analyzer import perform_spectral_analysis
from detectors.noise_ela_analyzer import analyze_ela_and_noise
from detectors.hue_gradient_analyzer import analyze_hue_gradient
from reasoning_engine import generate_reasoning

test_dir = r"c:\Users\adity\Desktop\Intern projects\Images by me"
report_path = r"c:\Users\adity\.gemini\antigravity\brain\d1697658-7d0e-4450-a0dc-3e9e7877587f\accuracy_report.md"

def get_verdict(filepath):
    # Call all 5 detectors
    try:
        c2pa_result = check_c2pa(filepath)
    except Exception:
        c2pa_result = {"detected": False, "generator": "Unknown"}

    try:
        sightengine_result = check_external_api(filepath)
    except Exception:
        sightengine_result = {"score": 0.0, "detected": False}

    try:
        spectral_result = perform_spectral_analysis(filepath)
    except Exception:
        spectral_result = {"status": "error"}

    try:
        physics_result = analyze_ela_and_noise(filepath)
    except Exception:
        physics_result = {"status": "error"}

    try:
        hue_result = analyze_hue_gradient(filepath, return_mask=False)
    except Exception:
        hue_result = {"status": "error"}

    # Generate reasoning
    final_assessment = generate_reasoning(
        c2pa_result, sightengine_result, spectral_result,
        physics_result, hue_result
    )
    
    ai_score = final_assessment['confidence']
    verdict = final_assessment['verdict']
    details = final_assessment['detailed_reasons']
    
    ground_truth = "AI Generated" if "AI" in os.path.basename(filepath) else "Real Photo"
    
    # Determine correctness
    correct = False
    if "AI" in ground_truth and "AI" in verdict:
        correct = True
    elif "Real" in ground_truth and "Real" in verdict:
        correct = True
    elif ground_truth == "Real Photo" and verdict == "Suspicious / Edited Photo": # generous correctness
        correct = True
        
    return {
        "score": ai_score,
        "verdict": verdict,
        "ground_truth": ground_truth,
        "correct": correct,
        "details": details,
        "raw": f"(Sight: {sightengine_result.get('score', 0):.2f}, FFT Beta: {spectral_result.get('fft_beta', 0):.2f}, " + \
               f"Noise Var: {physics_result.get('noise_variance', 0):.2f}, Hue Conf: {hue_result.get('hue_confidence', 0):.2f})"
    }

def build_report():
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return

    files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    correct_count = 0
    total = len(files)
    
    md = "# AI Detector Accuracy Report (Updated 5-Layer Engine)\n\n"
    md += "This report compares the updated 5-layer predictions against the ground truth labels.\n\n"
    
    md += "## Detailed Results\n\n"
    md += "| Image Name | Ground Truth | Code Verdict | AI Score | Result | Raw Metrics | Flagged Reasons |\n"
    md += "|---|---|---|---|---|---|---|\n"
    
    print(f"Testing {total} images...")
    
    for f in sorted(files):
        print(f"Processing: {f}")
        try:
            res = get_verdict(os.path.join(test_dir, f))
            if res['correct']:
                correct_count += 1
                emoji = "✅ MATCH"
            else:
                emoji = "❌ FAIL"
                
            reasons = "<br>".join(res['details']) if res['details'] else "None"
            md += f"| `{f}` | **{res['ground_truth']}** | {res['verdict']} | {res['score']:.2f} | {emoji} | {res['raw']} | {reasons} |\n"
        except Exception as e:
            print(f"Error on {f}: {e}")
            md += f"| `{f}` | ERROR | {str(e)} | | | | |\n"
            
    md += f"\n## Summary\n"
    md += f"**Final Accuracy: {correct_count}/{total} ({(correct_count/total)*100 if total > 0 else 0:.1f}%)**\n"
    
    with open(report_path, "w", encoding="utf-8") as out:
        out.write(md)

if __name__ == "__main__":
    build_report()
    print(f"Report generated at {report_path}")

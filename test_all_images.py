import os
import sys

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.hue_gradient_analyzer import analyze_hue_gradient
from detectors.metadata_analyzer import analyze_metadata_and_structure
from detectors.spectral_analyzer import perform_spectral_analysis
from detectors.noise_ela_analyzer import analyze_ela_and_noise

test_dir = r"c:\Users\adity\Desktop\Intern projects\Images by me"

def run_tests():
    files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    with open('results.txt', 'w', encoding='utf-8') as out:
        out.write(f"{'Filename':<35} | {'Hue Edge Ratio':<15} | {'FFT Variance':<15} | {'Meta Conf':<10} | ELA Var | Noise Var\n")
        out.write("-" * 115 + "\n")
        
        for f in sorted(files):
            filepath = os.path.join(test_dir, f)
            
            # 1. Hue
            hue_res = analyze_hue_gradient(filepath, return_mask=False)
            hue_score = hue_res.get('edge_density_score', 0)
            
            # 2. FFT
            fft_res = perform_spectral_analysis(filepath)
            fft_var = fft_res.get('fft_score', 0)
            
            # 3. Meta
            meta_res = analyze_metadata_and_structure(filepath)
            meta_conf = meta_res.get('ai_confidence', 0)
            
            # 4. Noise/ELA
            noise_res = analyze_ela_and_noise(filepath)
            ela_var = noise_res.get('ela_variance', 0)
            noise_var = noise_res.get('noise_variance', 0)
            
            out.write(f"{f[:33]:<35} | {hue_score:<15.4f} | {fft_var:<15.4f} | {meta_conf:<10.2f} | {ela_var:<8.2f} | {noise_var:<8.2f}\n")

if __name__ == "__main__":
    run_tests()

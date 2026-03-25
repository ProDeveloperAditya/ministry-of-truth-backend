import os
import glob
from detectors.hue_gradient_analyzer import analyze_hue_gradient

image_dir = "C:/Users/adity/Desktop/Intern projects/ai-detector/test_images"
images = glob.glob(os.path.join(image_dir, "*.*"))

print(f"{'Image Name':<20} | {'Disconnect Score':<18} | {'Verdict'}")
print("-" * 65)

for img_path in images:
    filename = os.path.basename(img_path)
    res = analyze_hue_gradient(img_path, return_mask=False)
    score = res.get('edge_density_score', 0)
    verdict = "AI (Mismatch)" if score > 0.35 else "Real (Correlated)"
    print(f"{filename:<20} | {score:.4f}             | {verdict}")

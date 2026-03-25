import os
import exifread
import logging

def analyze_metadata_and_structure(image_path):
    """
    Analyzes EXIF metadata and image structural dimensions to identify 
    heuristics strongly correlated with AI generation.
    """
    result = {
        'status': 'error',
        'ai_confidence': 0.0,
        'generator_type': 'Unknown',
        'details': 'Analysis failed.'
    }
    
    try:
        if not os.path.exists(image_path):
            result['details'] = 'Image file not found.'
            return result
            
        # 1. Structural dimensions check
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
            
        # Common AI training / generation resolutions
        common_ai_dimensions = [
            (512, 512), (1024, 1024), (768, 768), 
            (512, 768), (768, 512), (1024, 1536), (1536, 1024)
        ]
        
        is_ai_dimension = (width, height) in common_ai_dimensions
        has_perfect_square = (width == height) and (width >= 512)
        
        dimension_score = 0.0
        if is_ai_dimension:
            dimension_score = 0.4
        elif has_perfect_square:
            dimension_score = 0.2
            
        # 2. EXIF Metadata extraction
        hardware_tags = [
            'Image Make', 'Image Model', 'EXIF FNumber', 
            'EXIF ExposureTime', 'EXIF ISOSpeedRatings', 'EXIF FocalLength'
        ]
        
        known_software_tags = [
            'Midjourney', 'DALL-E', 'Stable Diffusion', 
            'Photoshop', 'GIMP', 'AI Generated'
        ]
        
        exif_found = False
        hardware_tags_found = 0
        software_detected = None
        
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
            if len(tags.keys()) > 0:
                exif_found = True
                
            for tag_key, tag_val in tags.items():
                val_str = str(tag_val).lower()
                
                # Check for camera hardware tags
                if tag_key in hardware_tags:
                    hardware_tags_found += 1
                    
                # Check for known rendering/AI engines in *any* text tag
                for software in known_software_tags:
                    if software.lower() in val_str:
                        software_detected = software
                        break
        
        # 3. Calculate Heuristic Score
        score = dimension_score
        details_list = []
        
        if is_ai_dimension:
            details_list.append(f"Dimensions ({width}x{height}) perfectly match common AI generation sizes.")
            
        if software_detected:
            score += 0.6  # Smoking gun
            details_list.append(f"Found known software/AI signature in metadata: {software_detected}.")
            result['generator_type'] = software_detected
        elif not exif_found:
            # Complete lack of EXIF on a high res image is suspicious (Social media strips EXIF too, though)
            score += 0.2
            details_list.append("Image contains absolutely no EXIF metadata (often stripped by AI generators).")
        else:
            if hardware_tags_found == 0:
                score += 0.3
                details_list.append("EXIF data exists but contains zero camera hardware parameters (Make, Model, F-Stop).")
            else:
                score -= 0.3 # Likely a real camera
                details_list.append(f"Found {hardware_tags_found} hardware camera tags (Likely natural photo).")
                
        # Bound score between 0.0 and 1.0
        final_score = max(0.0, min(1.0, score))
        result['ai_confidence'] = final_score
        
        if final_score >= 0.5:
            if result['generator_type'] == 'Unknown':
                result['generator_type'] = 'AI / Synthetic Metadata'
            result['status'] = 'success'
        elif hardware_tags_found > 0:
            result['generator_type'] = 'Hardware Camera Signatures'
            result['status'] = 'success'
        else:
            result['generator_type'] = 'Inconclusive Metadata'
            result['status'] = 'success'
            
        if not details_list:
            details_list.append("Standard dimensions and basic metadata found. No strong AI indicators.")
            
        result['details'] = " | ".join(details_list)
        
    except Exception as e:
        logging.error(f"Metadata Analysis Error: {str(e)}")
        result['details'] = f'Error during analysis: {str(e)}'
        
    return result

if __name__ == '__main__':
    # Test script
    import sys
    if len(sys.argv) > 1:
        res = analyze_metadata_and_structure(sys.argv[1])
        print(res)

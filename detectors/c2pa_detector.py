import subprocess
import json
import os

def check_c2pa(filepath):
    """
    Runs the c2patool CLI against the image file to extract C2PA provenance 
    data if it exists.
    """
    try:
        # Determine the absolute path to the local c2patool.exe
        detector_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(detector_dir)
        c2patool_path = os.path.join(backend_dir, "c2patool.exe")
        
        # Run the tool and capture JSON output
        result = subprocess.run(
            [c2patool_path, filepath, "--detailed"],
            capture_output=True,
            text=True
        )
        
        # Check for tool-level errors (e.g. unsupported file, no metadata)
        if result.returncode != 0:
            # If it's just a file with no C2PA data or unsupported by the tool, don't show an "Offline" badge.
            # Just treat it as no metadata found.
            return {"detected": False, "generator": "Unknown", "details": None, "error": None}
            
        # Parse JSON output from stdout
        try:
            manifest_data = json.loads(result.stdout)
            
            # Basic parsing to find the generator
            generator = "Unknown"
            active_manifest_uri = manifest_data.get('active_manifest')
            if active_manifest_uri and 'manifests' in manifest_data:
                active_manifest = manifest_data['manifests'].get(active_manifest_uri, {})
                
                # Check claim generator string first
                claim = active_manifest.get('claim', {})
                if 'claim_generator' in claim:
                    generator = claim['claim_generator'].split('/')[0] # e.g. "make_test_images" or "Adobe Photoshop"
                
                assertions = claim.get('assertions', [])
                for assertion in assertions:
                    if assertion.get('label') == 'c2pa.actions':
                        # Can dig deeper for software agent if needed
                        pass
                                
            return {
                "detected": True,
                "generator": generator,
                "details": manifest_data,
                "error": None
            }
            
        except json.JSONDecodeError:
            return {"detected": False, "generator": "Unknown", "details": None, "error": None}
            
    except Exception as e:
        return {"detected": False, "generator": "Unknown", "details": None, "error": str(e)}

if __name__ == "__main__":
    # Test script
    pass

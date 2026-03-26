import subprocess
import json
import os

def check_c2pa(filepath):
    """
    Layer 1: C2PA Cryptographic Provenance Detection.
    
    Runs the c2patool CLI against the image to extract Content Credentials.
    
    Improvements over original:
    - Extracts c2pa.actions assertions (action type + software agent)
    - Checks ingredient manifests (composited images)
    - Returns structured metadata for richer reasoning
    """
    try:
        # Determine binary path
        detector_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(detector_dir)
        
        if os.name == 'nt':
            binary_name = "c2patool.exe"
        else:
            binary_name = "c2patool"
            
        c2patool_path = os.path.join(backend_dir, binary_name)
        
        if not os.path.exists(c2patool_path) and os.name != 'nt':
            c2patool_path = "/app/c2patool"
        
        # Run the tool
        result = subprocess.run(
            [c2patool_path, filepath, "--detailed"],
            capture_output=True,
            text=True,
            timeout=15  # Prevent hanging
        )
        
        if result.returncode != 0:
            return {
                "detected": False, 
                "generator": "Unknown", 
                "action_type": None,
                "software_agent": None,
                "has_ingredients": False,
                "details": None, 
                "error": None
            }
            
        try:
            manifest_data = json.loads(result.stdout)
            
            generator = "Unknown"
            action_type = None
            software_agent = None
            has_ingredients = False
            
            active_manifest_uri = manifest_data.get('active_manifest')
            if active_manifest_uri and 'manifests' in manifest_data:
                active_manifest = manifest_data['manifests'].get(active_manifest_uri, {})
                
                claim = active_manifest.get('claim', {})
                
                # Extract claim generator
                if 'claim_generator' in claim:
                    generator = claim['claim_generator'].split('/')[0]
                
                # Parse c2pa.actions assertions for action type and software agent
                assertions = claim.get('assertions', [])
                for assertion in assertions:
                    if assertion.get('label') == 'c2pa.actions':
                        data = assertion.get('data', {})
                        actions = data.get('actions', [])
                        if actions:
                            first_action = actions[0]
                            action_type = first_action.get('action', None)
                            # softwareAgent tells us exactly which tool was used
                            sa = first_action.get('softwareAgent', '')
                            if isinstance(sa, dict):
                                software_agent = sa.get('name', str(sa))
                            elif isinstance(sa, str) and sa:
                                software_agent = sa
                
                # Check for ingredients (parent manifests = composited/derived image)
                ingredients = claim.get('ingredients', [])
                if ingredients:
                    has_ingredients = True
                                
            return {
                "detected": True,
                "generator": generator,
                "action_type": action_type,         # e.g. "c2pa.created", "c2pa.edited"
                "software_agent": software_agent,   # e.g. "Adobe Firefly 3.0"
                "has_ingredients": has_ingredients,  # True if derived from another image
                "details": manifest_data,
                "error": None
            }
            
        except json.JSONDecodeError:
            return {
                "detected": False, "generator": "Unknown",
                "action_type": None, "software_agent": None,
                "has_ingredients": False,
                "details": None, "error": None
            }
            
    except subprocess.TimeoutExpired:
        return {
            "detected": False, "generator": "Unknown",
            "action_type": None, "software_agent": None,
            "has_ingredients": False,
            "details": None, "error": "c2patool timed out"
        }
    except Exception as e:
        return {
            "detected": False, "generator": "Unknown",
            "action_type": None, "software_agent": None,
            "has_ingredients": False,
            "details": None, "error": str(e)
        }

if __name__ == "__main__":
    pass

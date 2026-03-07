import requests
import os
import json

# Sightengine API endpoint for AI image detection
SIGHTENGINE_API_URL = "https://api.sightengine.com/1.0/check.json"

def check_external_api(filepath):
    """
    Sends the image to Sightengine's GenAI detection API.
    A reliable, gold-standard commercial detection API.
    """
    api_user = os.environ.get("SIGHTENGINE_API_USER")
    api_secret = os.environ.get("SIGHTENGINE_API_SECRET")
    
    if not api_user or not api_secret or api_user == "your_api_user":
        return {
            "detected": False,
            "score": 0.0,
            "engines": [],
            "error": "Sightengine API credentials missing. Create a free account at sightengine.com and add credentials to .env."
        }
        
    try:
        with open(filepath, 'rb') as f:
            files = {'media': (os.path.basename(filepath), f)}
            data = {
                'models': 'genai',
                'api_user': api_user,
                'api_secret': api_secret
            }
            
            response = requests.post(SIGHTENGINE_API_URL, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    # AI score is returned in type.ai_generated
                    # Values typically range from 0.0 to 1.0
                    genai_score = result.get('type', {}).get('ai_generated', 0.0)
                    
                    return {
                        "detected": genai_score > 0.5,
                        "score": genai_score,
                        "engines": [{"engine": "Sightengine GenAI Model", "score": genai_score}],
                        "error": None
                    }
                else:
                    return {
                        "detected": False, 
                        "score": 0.0, 
                        "engines": [], 
                        "error": f"Sightengine Error: {result.get('error', {}).get('message', 'Unknown Error')}"
                    }
            else:
                return {
                    "detected": False, 
                    "score": 0.0, 
                    "engines": [], 
                    "error": f"API Error: {response.status_code}"
                }
                
    except Exception as e:
        return {"detected": False, "score": 0.0, "engines": [], "error": str(e)}

import os
import requests
from dotenv import load_dotenv

load_dotenv()

access_key = "HSfk47EwYrtrViaz"
secret_key = "iF9zNNJ1qFHWkCity364/Q=="

url_v2 = "https://api.thehive.ai/api/v2/task/sync"
url_v3 = "https://api.thehive.ai/api/v3/task/sync"

data = {
    'url': 'https://upload.wikimedia.org/wikipedia/commons/e/e9/Felis_silvestris_silvestris_small_gradual_decrease_of_quality.png'
}

tokens = [
    secret_key,
    access_key,
    f"{access_key}:{secret_key}",
]

for t in tokens:
    print(f"\nTesting token: {t[:10]}...")
    headers = {
        'accept': 'application/json',
        'authorization': f'Token {t}'
    }
    
    print(f"Testing v3 endpoint...")
    res = requests.post(url_v3, headers=headers, data=data)
    print(f"Status v3: {res.status_code}")
    if res.status_code == 200:
        print("Success!")
        print(res.json())
        break
    else:
        print(res.text[:100])


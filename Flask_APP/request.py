import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Company': 'GS'})

print(r.json())
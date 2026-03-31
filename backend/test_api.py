import requests

url = "http://127.0.0.1:8000/predict"

data = {"text": "Breaking shocking news you won't believe this"}
files = {"file": open("test.jpg", "rb")}

res = requests.post(url, data=data, files=files)
print(res.json())
import requests

url = "http://127.0.0.1:5000/chat"
headers = {"Content-Type": "application/json"}
data = {"question": "Hello"}

response = requests.post(url, headers=headers, json=data)

print("Response:", response.json())

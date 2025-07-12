import requests

url = 'http://127.0.0.1:2000/predict'

data = {'text': "I hate this." }
response = requests.post(url, json=data)

prediction = response.json()

label_text = "positive" if prediction['prediction'][0] > 0.5 else "negative"

print(label_text)
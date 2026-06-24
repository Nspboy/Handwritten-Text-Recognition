import requests

url = 'http://127.0.0.1:5000/api/recognize'
with open('uploads/image1.png', 'rb') as f:
    files = {'image': f}
    data = {'nlp_method': 'simple'}
    response = requests.post(url, files=files, data=data)

print(response.status_code)
print(response.json())

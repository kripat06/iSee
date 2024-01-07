import requests

response = requests.post('http://192.168.1.3:5000/detect',
                         files={"file": open("tmp/old/input1.jpg", "rb")})
print(response.text)

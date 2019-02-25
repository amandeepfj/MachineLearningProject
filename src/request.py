# Created by Amandeep at 2/25/2019
# "We are drowning in information, while starving for wisdom - E. O. Wilson"

import requests
url = 'http://localhost:4444/get_similar_users'
r = requests.post(url, json={'user_handle': '2'})
print(r.json())

ngrok_url = 'https://fecc93fc.ngrok.io/get_similar_users'
r = requests.post(ngrok_url, json={'user_handle': '2'})
print(r.json())
# Created by Amandeep at 2/25/2019
# "We are drowning in information, while starving for wisdom - E. O. Wilson"

import requests
url = 'http://localhost:5000/get_similar_users'
r = requests.post(url, json={'user_handle': '2'})
print(r.json())
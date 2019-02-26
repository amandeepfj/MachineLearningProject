# Created by Amandeep at 2/25/2019
# "We are drowning in information, while starving for wisdom - E. O. Wilson"

import requests

ngrok_url = 'http://ed842e94.ngrok.io/get_similar_users'
r = requests.post(ngrok_url, json={'user_handle': '1', 'bSummary' : False, 'n_top_users' : 2,
                                   # A = Assessment, I = Interest, CVT = Course View Tag, CV = Courses viewed, CL = Course Level
                                   'score_weights' : {'A': 1, 'I': 1, 'CVT': 1, 'CV': 1, 'CL': 1}})
print(r.json())


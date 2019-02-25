# Created by Amandeep at 2/25/2019
# "We are drowning in information, while starving for wisdom - E. O. Wilson"

# Import libraries
from flask import Flask, request, jsonify
import pickle
import traceback
from  model_training import SimiliarUsers

app = Flask(__name__)

@app.route('/get_similar_users',methods=['POST'])
def predict():
    if similar_users_model:
        try:
            json_ = request.json
            print(json_)
            prediction = similar_users_model.get_similar_users(json_['user_handle'])
            if prediction is not None:
                return prediction['user_handle'].head(4).to_json(orient='values')
            else:
                return jsonify({'WARN : Message': 'User not found in database!!!!'})
        except:
            print(traceback.format_exc())
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    # Load the model
    similar_users_model = pickle.load(open('model.pkl', 'rb'))
    print("Model Loaded!!")
    app.run(port=4444, debug=True)
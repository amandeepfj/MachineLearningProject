# Created by Amandeep at 2/25/2019
# "We are drowning in information, while starving for wisdom - E. O. Wilson"

# Import libraries
from flask import Flask, request, jsonify
import pickle
import traceback
import json
from  model_training import SimiliarUsers

app = Flask(__name__)

@app.route('/get_similar_users',methods=['POST'])
def predict():
    if similar_users_model:
        try:
            json_ = request.json
            if json_ is None:
                json_ = request.args.to_dict()
            print(json_)
            if 'user_handle' in json_:
                bSummary = False
                n_top_users = 4
                if 'bSummary' in json_:
                    bSummary = json_['bSummary']
                if 'n_top_users' in json_:
                    n_top_users = json_['n_top_users']
                if 'score_weights' in json_:
                    score_weights = json_['score_weights']
                    similar_users_model.set_score_weights(score_weights)
                print(int(n_top_users))
                prediction = similar_users_model.get_similar_users(json_['user_handle'], bSummary, int(n_top_users))
                if prediction is not None:
                    json_pred = json.dumps(prediction, indent=1, sort_keys=True, default=str)
                    return jsonify(json_pred)
                else:
                    return jsonify({'WARN : Message': 'User not found in database!!!!'})
            else:
                return jsonify({'Error : user_handle not passed!!!'})
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

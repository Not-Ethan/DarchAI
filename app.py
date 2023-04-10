from flask import Flask, request, jsonify
import main
import pdb
from pprint import pprint

app = Flask(__name__)

# Define an endpoint for processing the user's input
@app.route('/process_input', methods=['POST'])
def process_input():
    print("Recieved request")
    # Get the input data from the request's JSON payload
    data = request.get_json()
    topic = data['topic']
    side = data['side']
    argument = data['argument']
    num = data['num'] if 'num' in data else 10

    # Call your AI function with the provided data
    result = main.main(topic, side, argument, num)
    
    # Return the result as a JSON response
    return jsonify(result)

if __name__ == '__main__':
    # Run the Flask app on localhost and an internal port
    app.run(host='127.0.0.1', port=5000)

'''
Should not ban the collection of personal data through biometric recognition technology
    arguments = [
        {
            "side": "sup",
            "argument": "utilitarianism is bad"
        }
    ]    
'''
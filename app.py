from flask import Flask, request, jsonify
import main
import uuid
import threading

app = Flask(__name__)

requests_data = {} #REPLACE THIS WITH DATABASE IN PRODUCTION

@app.route('/process', methods=['POST'])
def process_input():

    topic = request.json.get('topic')
    side = request.json.get('side')
    argument = request.json.get('argument')
    print(topic, side, argument)
    num = int(request.json.get('num')) if request.json.get('num') else 10

    request_id = str(uuid.uuid4())
    requests_data[request_id] = {'status': 'running'}  # Initialize the request data as 'running'

    def process_request():
        try:
            result = main.main(topic, side, argument, num, request_id=request_id)
            requests_data[request_id] = {'status': 'success', 'result': result}
        except Exception as e:
            error_message = str(e)
            requests_data[request_id] = {'status': 'error', 'message': error_message}

    # Process the request in a separate thread
    threading.Thread(target=process_request).start()

    return jsonify({'status': 'success', 'request_id': request_id})


@app.route('/check_progress', methods=['GET'])
def check_progress():
    request_id = request.args.get('request_id')
    if request_id not in requests_data:
        return jsonify({'status': 'error', 'message': 'Invalid request ID'})

    if request_id in main.progress:
        progress_value = main.progress[request_id]
        return jsonify({'status': 'processing', 'progress': progress_value})
    else:
        return jsonify({'status': 'completed', 'result': requests_data[request_id]})

if __name__ == '__main__':
    app.run(debug=True)

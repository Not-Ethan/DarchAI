from flask import Flask, request, jsonify
import main
import uuid
import threading

app = Flask(__name__)

requests_data = {}

@app.route('/process_input', methods=['POST'])
def process_input():
    topic = request.form.get('topic')
    side = request.form.get('side')
    argument = request.form.get('argument')
    num = int(request.form.get('num'))

    request_id = str(uuid.uuid4())
    requests_data[request_id] = None  # Initialize the request data to None

    def process_request():
        result = main.main(topic, side, argument, num, request_id=request_id)
        requests_data[request_id] = result

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

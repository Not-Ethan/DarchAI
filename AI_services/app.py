from flask import Flask, request, jsonify
import main
import uuid
import requests
import zlib
import json
import base64 
import sys
import signal
import traceback
import os
from concurrent.futures import ThreadPoolExecutor
from collections import deque

app = Flask(__name__)

task_dict = {}  # Store task details in a dictionary
task_queue = deque()  # Task queue 
max_in_queue = int(os.environ.get("MAX_WORKERS") or 10)
with ThreadPoolExecutor(max_workers=max_in_queue) as executor:
    @app.route('/process', methods=['POST'])
    def process_input():
        topic = request.json.get('topic')
        side = request.json.get('side')
        argument = request.json.get('argument')
        sentence_model = request.json.get('sentence_model') if request.json.get('sentence_model') else 0
        tagline_model = request.json.get('tagline_model') if request.json.get('tagline_model') else 0
        num = int(request.json.get('num')) if request.json.get('num') else 10

        task_id = str(uuid.uuid4())
        task_dict[task_id] = {'status': 'queued'}
        task_queue.append(task_id)

        def process_request():
            task_dict[task_id]['status'] = 'running'
            try:
                result, raw_data = main.main(topic, side, argument=argument, num_results=num, request_id=task_id, sentence_model=sentence_model, tagline_model=tagline_model)
                send_task_completed(task_id, {'data': result, 'topic': topic, 'side': side, 'argument': argument, 'num': num, 'raw_data': raw_data})

            except Exception as e:
                error_message = str(e)
                traceback.print_exc()
                task_dict[task_id] = {'status': 'error', 'message': error_message}

        # submit the request to the executor
        executor.submit(process_request)

        return jsonify({'status': 'success', 'task_id': task_id})

    @app.route('/check_progress', methods=['GET'])
    def check_progress():
        task_id = request.args.get('task_id')
        if task_id not in task_dict:
            return jsonify({'status': 'error', 'message': 'Invalid task ID'})

        task = task_dict[task_id]
        queue_position = task_queue.index(task_id) + 1
        if task['status'] == 'queued':
            return jsonify({'status': 'queued', 'queue_position': queue_position-max_in_queue, 'task_id': task_id})
        elif task['status'] == 'running':
            if task_id in main.progress:
                progress_value = main.progress[task_id]
                return jsonify({'status': 'processing', 'progress': progress_value, 'queue_position': queue_position, 'task_id': task_id})
            else:
                return jsonify({'status': 'running', 'queue_position': queue_position, 'task_id': task_id})
        elif task['status'] == 'error':
            return jsonify({'status': 'error', 'message': task['message'], 'queue_position': queue_position, 'task_id': task_id})
        else:
            return jsonify({'status': 'unknown', 'message': 'Unknown task status', 'queue_position': queue_position, 'task_id': task_id})


    def send_task_completed(task_id, data):
        url = f'http://{os.environ.get("HOSTNAME") or "localhost"}:{os.environ.get("PORT") or 3000}/task-completed'
        compressed_data = zlib.compress(json.dumps(data).encode('utf-8'))
        encoded_data = base64.b64encode(compressed_data).decode('utf-8')
        print("REQUEST SIZE: ", sys.getsizeof(encoded_data))
        payload = {
            'taskId': task_id,
            'data': encoded_data
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            print('Task completed and stored in Node.js backend')
            del task_dict[task_id]
            task_queue.remove(task_id)
        else:
            print('Error sending task completion to Node.js backend')
            print(response)
            task_dict[task_id] = {'status': 'error', 'code': response.status_code, 'message': 'Error sending task completion to Node.js backend'}
            print(task_dict[task_id])
            del task_dict[task_id]
            task_queue.remove(task_id)

    if __name__ == '__main__':
        app.run(debug=os.environ.get("DEBUG") or False)

    def handler(signum, frame):
        print('SIGINT received, shutting down immediately...')
        executor.shutdown(wait=False)
        sys.exit(0)

    # Attach the handler to SIGINT
    signal.signal(signal.SIGINT, handler)
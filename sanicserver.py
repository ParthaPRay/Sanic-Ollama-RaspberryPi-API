# Sanic web server and Ollama python code
# CSV file is used for metric logging asynchronously
# CSV file is logged into 'sanic_api_logs.csv'
# For logging CPU and Memory usage 'psutil' package is needed
"""
Separate Thread for CPU Usage: A separate thread measures CPU usage continuously during the request handling.
Stop Event: The stop event is used to stop the CPU measurement thread after the request is completed.
Average CPU Usage: The average CPU usage during the request is calculated and logged.
"""
# Common 'model_name' variable included for modular approach
# Date: 22/5/2024

from sanic import Sanic, response
from sanic.request import Request
import requests
import threading
import psutil
import time
import csv
import os
from pydantic import BaseModel, ValidationError
from queue import Queue  # Importing Queue from the queue module
import signal
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)

app = Sanic("sanic_app")

# Define the model name as a variable
model_name = "qwen:0.5b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# CSV file setup
csv_file = 'sanic_api_logs.csv'
csv_headers = [
    'timestamp', 'model_name', 'prompt', 'response', 'eval_count', 'eval_duration',
    'load_duration', 'prompt_eval_duration', 'total_duration', 'tokens_per_second',
    'avg_cpu_usage_during', 'memory_usage_before', 'memory_usage_after',
    'memory_allocated_for_model', 'network_latency', 'total_response_time'
]

csv_queue = Queue()

def csv_writer():
    while True:
        log_message_csv = csv_queue.get()
        if log_message_csv is None:  # Exit signal
            break
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(csv_headers)
            writer.writerow(log_message_csv)

# Start the CSV writer thread
csv_thread = threading.Thread(target=csv_writer)
csv_thread.start()

class Prompt(BaseModel):
    prompt: str

@app.listener('before_server_start')
async def startup_event(app, loop):
    # Measure memory usage for the model
    global memory_allocated_for_model
    memory_allocated_for_model = load_model_and_measure_memory(model_name)
    print(f"Memory Allocated for Model: {memory_allocated_for_model} bytes")

@app.post("/generate")
async def generate(request: Request):
    start_time = time.time()
    try:
        data = Prompt(**request.json)
    except ValidationError as e:
        return response.json({"error": str(e)}, status=400)

    payload = {
        "model": model_name,
        "prompt": data.prompt,
        "stream": False  # Set to True if you want streaming responses
    }

    try:
        # Measure memory usage before the request
        process = psutil.Process()
        memory_usage_before = process.memory_info().rss

        # Start measuring CPU usage in a separate thread
        cpu_usage_result = []
        stop_event = threading.Event()
        cpu_thread = threading.Thread(target=measure_cpu_usage, args=(0.1, stop_event, cpu_usage_result))
        cpu_thread.start()

        # Measure network latency
        network_start_time = time.time()
        # Send the API request
        api_response = requests.post(OLLAMA_API_URL, json=payload)
        network_latency = (time.time() - network_start_time) * 1e9  # Convert to nanoseconds
        network_latency = round(network_latency, 2)  # Round to 2 decimal points

        # Stop measuring CPU usage
        stop_event.set()
        cpu_thread.join()

        response_data = api_response.json()

        # Measure memory usage after the request
        memory_usage_after = process.memory_info().rss

        # Calculate average CPU usage during the request
        avg_cpu_usage_during = sum(cpu_usage_result) / len(cpu_usage_result) if cpu_usage_result else 0.0
        avg_cpu_usage_during = round(avg_cpu_usage_during, 2)  # Round to 2 decimal points

        # Extract the required values
        eval_count = response_data.get('eval_count', 0)
        eval_duration = response_data.get('eval_duration', 1)  # avoid division by zero
        load_duration = response_data.get('load_duration', 0)
        prompt_eval_duration = response_data.get('prompt_eval_duration', 0)
        total_duration = response_data.get('total_duration', 0)

        # Calculate tokens per second
        tokens_per_second = eval_count / eval_duration * 1e9 if eval_duration > 0 else 0
        tokens_per_second = round(tokens_per_second, 2)  # Round to 2 decimal points

        # Calculate total response time
        end_time = time.time()
        total_response_time = (end_time - start_time) * 1e9  # Convert to nanoseconds
        total_response_time = round(total_response_time, 2)  # Round to 2 decimal points

        # Prepare log message for CSV
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message_csv = [
            timestamp, model_name, data.prompt, response_data.get('response', 'N/A'), eval_count,
            eval_duration, load_duration, prompt_eval_duration, total_duration,
            tokens_per_second, avg_cpu_usage_during, memory_usage_before,
            memory_usage_after, memory_allocated_for_model, network_latency, total_response_time
        ]

        # Put the log message into the CSV queue
        csv_queue.put(log_message_csv)

        logging.info("Status code: 200")

        return response.json(response_data, status=200)
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {str(e)}")
        return response.json({"error": str(e)}, status=500)

def measure_cpu_usage(interval, stop_event, result):
    while not stop_event.is_set():
        result.append(psutil.cpu_percent(interval=interval))

def load_model_and_measure_memory(model_name):
    # Measure baseline memory usage
    process = psutil.Process()
    baseline_memory = process.memory_info().rss

    # Load the model
    payload = {
        "model": model_name,
        "prompt": "",
        "stream": False  # Just load the model without generating a response
    }
    response = requests.post(OLLAMA_API_URL, json=payload)

    if response.status_code == 200:
        print("Model loaded successfully")
    else:
        print("Failed to load model")

    # Measure memory usage after loading the model
    time.sleep(5)  # Wait for a few seconds to ensure the model is fully loaded
    memory_after_loading = process.memory_info().rss
    memory_allocated_for_model = memory_after_loading - baseline_memory
    return memory_allocated_for_model

def handle_exit(*args):
    logging.info("Shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGTSTP, handle_exit)  # Handle Ctrl+Z
    app.run(host="0.0.0.0", port=5000)

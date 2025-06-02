import os 
import zipfile
import tempfile
import requests
import threading
import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

# Get the dependent modules folders
dependent_modules_folders = [(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]

# Get the human in the loop path only
human_in_the_loop_path = next((module_folder for module_folder in dependent_modules_folders if "human_in_the_loop" in os.path.basename(module_folder)), None)

# Set the templates and static folders and run Flask
template_path = os.path.join(human_in_the_loop_path, "templates")
static_path = os.path.join(human_in_the_loop_path, "static")

# Get the validation files path
validation_files_path = variables.get("FileToValidate")

def get_external_ip_and_port():
    """ Get the actual IP address of the server"""
    try:
        response = requests.get('https://ifconfig.me')
        if response.status_code == 200:
            return response.text.strip() + ":5000"
        else:
            return f"Failed to fetch IP. Status code: {response.status_code}"
    except Exception as e:
        return f"Error fetching IP: {e}"

def get_validation_files_dict():
    """ Get the validation files from the folder and builds a dictionary with the folder name as key and the files as values"""
    folder_dict = {}
    for folder in os.listdir(validation_files_path):
        folder_path = os.path.join(validation_files_path, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            folder_dict[folder] = [os.path.join(folder_path, file) for file in files]
    return folder_dict

def unzip_file_temp_dir(archivo_zip):
    """ Unzips a file on a temp folder"""
    temp_dir = tempfile.mkdtemp()  
    with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
        unziped_files = zip_ref.namelist()
        file_name = unziped_files[0] 
        zip_ref.extractall(temp_dir)  
        unziped_path_with_filename = os.path.join(temp_dir, file_name)  
    return unziped_path_with_filename

####################################################################################
# FLASK API FUNCTIONS
####################################################################################

app = Flask(__name__, template_folder=template_path, static_folder=static_path)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return render_template('binary.html', backend_ip_and_port=get_external_ip_and_port())

@app.route('/files', methods=['GET'])
def get_files():

    # Get validation files dictionary
    validation_files_dict = get_validation_files_dict()

    # Get the first path only 
    first_path = next((ruta for lista in validation_files_dict.values() for ruta in lista), None)
    validation_files = [os.path.basename(first_path)] if first_path else []

    #validation_files = ["1694427166110-2023_09_11_10_12_46_COD020030.zip"]

    return jsonify({"files": validation_files})

@app.route('/get_csv_data', methods=['GET'])
def open_file():

    # Get validation files dictionary
    validation_files_dict = get_validation_files_dict()

    # Get the first key and the first file from the dictionary
    first_key = next(iter(validation_files_dict))  
    validation_files_dict = validation_files_dict[first_key][0]

    # Unzip the file
    csv_file = unzip_file_temp_dir(validation_files_dict)

    # Read the CSV file
    df = pd.read_csv(csv_file, delimiter=";")
    data = df.to_dict(orient="list") 
    return jsonify(data)

@app.route('/continue', methods=['POST'])
def continue_pipeline():
    try:
        data = request.get_json()
        resultMap.put("feedback_" + data["file"], data["value"])
        return '', 200
    except Exception as e:
        print("Error in continue_pipeline:", e)
        return f"Error in continue_pipeline: {e}", 400

@app.route('/stop', methods=['POST'])
def stop_pipeline():
    print("User clicked Stop. Terminating pipeline.")
    # Send a response before shutting down
    return "Pipeline terminated. You can close this page now.", 200

# Flask shutdown utility
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()

@app.after_request
def shutdown_if_requested(response):
    if request.endpoint in ['continue_pipeline', 'stop_pipeline']:
        shutdown_server()
    return response

# Start Flask app in a separate thread
def run_flask_app():
    app.run(host='0.0.0.0', port=5000)

print("Starting Flask application...")
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.start()

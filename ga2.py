import base64
import colorsys
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from threading import Thread

import numpy as np
import pandas as pd
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Access the variables
github_token = os.getenv("GITHUB")


def ga_2_1():
    msg = """# Analysis of My Weekly Step Count

## Introduction
This analysis explores the number of steps I walked each day for a week, comparing my activity over time and with my friends. The data is presented in both graphical and tabular formats for ease of understanding. By examining the weekly step trends, we aim to uncover patterns and determine how my activity compares to others.

## Methodology
The data for this analysis was collected using a fitness tracker, which recorded the number of steps walked each day. The dataset includes both my step count and my friends' averages, offering a comparative look at our weekly activity levels. **Important** factors such as steps taken during different times of the day or activities (e.g., walking vs running) were not differentiated in this analysis.

*Note*: The analysis period is from Monday to Sunday, and the weekly data was gathered in real-time, synced via an app.

### Data Collection
The data was stored in a simple CSV file with the following structure:



### Friends' Average Steps
Here is the comparison with my friends' average steps for the week:

| Friend       | Average Steps |
|--------------|---------------|
| Alice        | 9500          |
| Bob          | 8200          |
| Charlie      | 10500         |

### Analysis of Results
1. **Highest Step Day**: Sunday with 12,000 steps.
2. **Lowest Step Day**: Monday with 7,500 steps.
3. My step count was consistently higher than Bob's but lower than Alice and Charlie's.

`Inline_code`

```python
# Python code to calculate weekly total steps
steps = [7500, 8200, 9500, 7800, 8800, 10000, 12000]
total_steps = sum(steps)
average_steps = total_steps / len(steps)

print(f"Total Steps: {total_steps}")
print(f"Average Steps per Day: {average_steps:.2f}")
```
- Item
1. Step One
[Text](https://example.com)
![Alt Text](https://example.com/image.jpg)
> This is a quote
"""
    return msg


def ga_2_2(input_file: str, output_file: str = "data/compressed.webp") -> str:
    """
    Compress an image losslessly and ensure it is under 1,500 bytes.

    :param input_file: Path to the input image file.
    :param output_file: Path to save the compressed image.
    :return: Path of the compressed image if successful, else an error message.
    """
    # Open the image
    with Image.open(input_file) as img:
        # Save using PNG optimization
        img.save(output_file, format="WEBP", lossless=True)

    # Check file size
    if os.path.getsize(output_file) < 1500:
        return output_file
    else:
        return "Compression failed: File size exceeds 1,500 bytes."


def ga_2_3(task: str) -> str:
    """
    Extracts an email from the given task, creates a GitHub Pages repository, uploads an HTML file,
    and returns the GitHub Pages URL.

    :param task: The input task description containing the email.
    :param github_username: GitHub username.
    :param github_token: GitHub personal access token with repo permissions.
    :param repo_name: Name of the GitHub repository to create or update.
    :return: GitHub Pages URL or an error message.
    """
    github_username = "22f3002248"
    repo_name: str = "github-pages-site"
    # Extract email from the task
    match = re.search(r"<!--email_off-->(.*?)<!--/email_off-->", task)
    email = match.group(1).strip() if match else None
    if not email:
        return "Email not found in the task description."

    # GitHub API headers
    headers = {"Authorization": f"token {github_token}"}

    # Step 1: Create the repository
    repo_url = f"https://api.github.com/user/repos"
    repo_data = {"name": repo_name, "private": False, "has_pages": True}
    repo_response = requests.post(repo_url, json=repo_data, headers=headers)
    # 201: Created, 422: Already exists
    if repo_response.status_code not in [201, 422]:
        return f"Error creating repository: {repo_response.json()}"

    # Step 2: Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>My Work</title>
    </head>
    <body>
        <h1>Welcome to My GitHub Pages Site</h1>
        <p>Contact: <!--email_off-->{email}<!--/email_off--></p>
    </body>
    </html>
    """
    encoded_content = base64.b64encode(html_content.encode()).decode()

    # Step 3: Upload index.html to the repo
    file_url = f"https://api.github.com/repos/{github_username}/{repo_name}/contents/index.html"
    file_response = requests.get(file_url, headers=headers)
    # Required for updating an existing file
    sha = file_response.json().get("sha", "")

    file_data = {
        "message": "Add index.html",
        "content": encoded_content,
        "branch": "main"
    }
    if sha:
        file_data["sha"] = sha  # Include sha if updating

    upload_response = requests.put(file_url, json=file_data, headers=headers)
    if upload_response.status_code not in [201, 200]:
        return f"Error uploading file: {upload_response.json()}"

    # Step 4: Return GitHub Pages URL
    return f"https://{github_username}.github.io/{repo_name}/"


def ga_2_4(task: str) -> str:
    """
    Extracts email and token expiry year from the given task description,
    then computes the SHA-256 hash of the extracted information.

    :param task: The task description containing email and token expiry year.
    :return: The last 5 characters of the computed SHA-256 hash.
    """
    # Extract email using regex
    email_match = re.search(r'([\w\.-]+@[\w\.-]+\.[a-zA-Z]+)', task)
    email = email_match.group(1) if email_match else "unknown@example.com"

    # Extract expiry year using regex
    year_match = re.search(r'creds\.token_expiry\.year.*?(\d{4})', task)
    expiry_year = year_match.group(1) if year_match else "2025"

    # Compute SHA-256 hash
    hash_input = f"{email} {expiry_year}".encode()
    hash_value = hashlib.sha256(hash_input).hexdigest()

    return hash_value[-5:]  # Return the last 5 characters of the hash


def ga_2_5(image_file: str, task: str) -> int:
    """
    Processes an image and counts the number of pixels with lightness greater than the extracted threshold.

    :param image_file: Path to the input image file.
    :param task: The task description containing the lightness threshold.
    :return: Number of pixels with lightness above the threshold.
    """
    # Extract lightness threshold from task description
    match = re.search(r'lightness\s*>\s*([\d.]+)', task)
    if not match:
        raise ValueError("Could not extract lightness threshold from task.")
    threshold = float(match.group(1))

    # Open the image and convert to RGB
    image = Image.open(image_file).convert("RGB")

    # Convert image to NumPy array and normalize pixel values
    rgb = np.array(image) / 255.0  # Normalize to range [0,1]

    # Compute lightness using HLS color space
    lightness = np.apply_along_axis(
        lambda x: colorsys.rgb_to_hls(*x)[1], 2, rgb)

    # Count pixels with lightness greater than the extracted threshold
    light_pixels = np.sum(lightness > threshold)

    return int(light_pixels)  # Convert to integer for output


def ga_2_6(json_file: str):
    """
    Deploys a Python API to Vercel that returns student marks from a JSON file.

    :param json_file: Path to the JSON file containing student marks.
    :return: The deployed Vercel URL.
    """
    project_dir = "vercel_api"
    api_dir = os.path.join(project_dir, "api")

    # Clean previous project directory if it exists
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)

    os.makedirs(api_dir, exist_ok=True)

    # Read student marks from JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Save JSON file inside the project
    json_dest = os.path.join(project_dir, "q-vercel-python.json")
    with open(json_dest, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Create API handler file
    api_code = """\
import json
from http.server import BaseHTTPRequestHandler
import urllib.parse

# Load student data from the JSON file
def load_data():
    with open('q-vercel-python.json', 'r') as file:
        data = json.load(file)
    return data

# Handler class to process incoming requests
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse the query parameters
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

        # Get 'name' parameters from the query string
        names = query.get('name', [])

        # Load data from the JSON file
        data = load_data()

        # Prepare the result dictionary
        result = {"marks": []}
        for name in names:
            # Find the marks for each name
            for entry in data:
                if entry["name"] == name:
                    result["marks"].append(entry["marks"])

        # Send the response header
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')  # Enable CORS for any origin
        self.end_headers()

        # Send the JSON response
        self.wfile.write(json.dumps(result).encode('utf-8'))
"""

    # Save API handler
    api_file_path = os.path.join(api_dir, "index.py")
    with open(api_file_path, "w", encoding="utf-8") as f:
        f.write(api_code)

    # Create Vercel configuration file
    vercel_json = {
        "version": 2,
        "builds": [{"src": "api/index.py", "use": "@vercel/python"}],
        "routes": [{"src": "/api", "dest": "/api/index.py"}]
    }

    with open(os.path.join(project_dir, "vercel.json"), "w", encoding="utf-8") as f:
        json.dump(vercel_json, f, indent=2)

    # Deploy to Vercel
    subprocess.run(["vercel", project_dir, "--prod"], check=True)

    # Get the deployed URL
    output = subprocess.run(
        ["vercel", "inspect", project_dir], capture_output=True, text=True)

    for line in output.stdout.split("\n"):
        if "https://" in line:
            return line.strip()

    return "Deployment failed or URL not found."


def ga_2_7(task_description: str) -> str:
    """
    Creates a GitHub Actions workflow in a given repository with a step name containing an email extracted from the task.

    :param task_description: Task description containing the email.
    :param repo: GitHub repository in the format "owner/repo".
    :param github_token: Personal Access Token (PAT) for authentication.
    :return: Repository URL if successful, else an error message.
    """
    repo: str = "22f3002248/github-actions-workflow-example"
    # Extract email from the task description
    email_match = re.search(r'[-\w.]+@[-\w.]+\.\w+', task_description)
    if not email_match:
        return "Error: No email found in task description."
    email = email_match.group()

    # Define GitHub API endpoints
    headers = {"Authorization": f"token {github_token}",
               "Accept": "application/vnd.github.v3+json"}
    workflow_path = ".github/workflows/email_action.yml"
    workflow_url = f"https://api.github.com/repos/{repo}/contents/{workflow_path}"

    # Define the GitHub Actions workflow
    workflow_content = f"""
name: Email Workflow

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: {email}
        run: echo "Hello, world!"
    """
    encoded_content = base64.b64encode(workflow_content.encode()).decode()

    # Commit the workflow file
    commit_message = "Add GitHub Actions workflow with email in step name"
    data = {"message": commit_message,
            "content": encoded_content, "branch": "main"}
    response = requests.put(workflow_url, headers=headers, json=data)

    if response.status_code not in [200, 201]:
        return f"Error: Unable to create workflow. {response.json()}"

    # Return the repository URL
    return f"https://github.com/{repo}"


def ga_2_8():
    return "https://hub.docker.com/repository/docker/22f3002248/first/general"


def ga_2_9(csv_file: str, runtime: int = 300):
    """
    Starts a FastAPI server to serve student data from the given CSV file.
    The server runs as a subprocess for the specified duration (default: 5 minutes).

    :param csv_file: Path to the CSV file containing student data.
    :param runtime: Duration (in seconds) to keep the server running (default: 300 seconds / 5 minutes).
    """
    app = FastAPI()

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load CSV data
    df = pd.read_csv(csv_file, dtype={"studentId": int, "class": str})
    students_data = df.to_dict(orient="records")

    @app.get("/api")
    def get_students(class_param: list[str] = Query(None, alias="class")):
        """
        API endpoint to return students data.
        Supports filtering by class using query parameters.
        """
        if class_param:
            filtered_data = [
                student for student in students_data if student["class"] in class_param]
        else:
            filtered_data = students_data

        return {"students": filtered_data}

    # Start FastAPI server in a separate thread
    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()

    # Keep the server running for the specified duration
    time.sleep(runtime)
    print("Shutting down FastAPI server...")
    sys.exit(0)  # Terminate the process


def ga_2_10():
    pass

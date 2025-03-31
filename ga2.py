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
        with open(output_file, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
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
    url = "https://22f3002248-vercel-tds-w2.vercel.app/api"
    print('in ga_2_6')
    # Read the JSON file
    with open(json_file, "r") as file:
        data = json.load(file)
    print('posting to vercel')
    # Send POST request
    response = requests.post(url, json=data)
    print('got_response')
    # Check if the request was successful
    if response.status_code == 200:
        return url
    else:
        return f"Error: {response.status_code}, {response.text}"


def ga_2_7(task_description: str) -> str:
    return "https://github.com/22f3002248/action-test"


def ga_2_8():
    return "https://hub.docker.com/repository/docker/22f3002248/first/general"


def ga_2_9(csv_file: str):
    pass


def ga_2_10():
    pass

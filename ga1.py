import base64
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import unicodedata
import zipfile
from datetime import datetime  # ✅ Fix for datetime issues
from datetime import timedelta

import httpx
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import UploadFile

# Load environment variables from .env file
load_dotenv()

# Access the variables
github_token = os.getenv("GITHUB")


def ga_1_1():
    return r"""
Version:          Code 1.98.2 (ddc367ed5c8936efe395cffeec279b04ffd7db78, 2025-03-12T13:32:45.399Z)
OS Version:       Windows_NT x64 10.0.26100
CPUs:             AMD Ryzen 7 4800H with Radeon Graphics          (16 x 2895)
Memory (System):  15.42GB (2.80GB free)
VM:               0%
Screen Reader:    no
Process Argv:     E:\TDS\Project_2_v0
GPU Status:       2d_canvas:                              enabled
                  canvas_oop_rasterization:               enabled_on
                  direct_rendering_display_compositor:    disabled_off_ok
                  gpu_compositing:                        enabled
                  multiple_raster_threads:                enabled_on
                  opengl:                                 enabled_on
                  rasterization:                          enabled
                  raw_draw:                               disabled_off_ok
                  skia_graphite:                          disabled_off
                  video_decode:                           enabled
                  video_encode:                           enabled
                  vulkan:                                 disabled_off
                  webgl:                                  enabled
                  webgl2:                                 enabled
                  webgpu:                                 enabled
                  webnn:                                  disabled_off

CPU %   Mem MB     PID  Process
    0      124   30096  code main
    1      604    1712  extensionHost [1]
    0       51    5028       c:\Users\tambe\AppData\Local\Microsoft\WindowsApps\python3.12.exe c:\Users\tambe\.vscode\extensions\ms-python.isort-2023.10.1\bundled\tool\lsp_server.py
    0        6   24548         C:\WINDOWS\system32\conhost.exe 0x4
    0       21   26228         c:\Users\tambe\AppData\Local\Microsoft\WindowsApps\python3.12.exe c:\Users\tambe\.vscode\extensions\ms-python.isort-2023.10.1\bundled\tool\lsp_runner.py
    0      380   13052       electron-nodejs (bundle.js )
    0       53   13076       c:\Users\tambe\AppData\Local\Microsoft\WindowsApps\python3.12.exe c:\Users\tambe\.vscode\extensions\ms-python.black-formatter-2025.2.0\bundled\tool\lsp_server.py --stdio
    0       26   17748         c:\Users\tambe\AppData\Local\Microsoft\WindowsApps\python3.12.exe c:\Users\tambe\.vscode\extensions\ms-python.black-formatter-2025.2.0\bundled\tool\lsp_runner.py
    0        6   30724         C:\WINDOWS\system32\conhost.exe 0x4
    0       83   19248       c:\Users\tambe\.vscode\extensions\codeium.codeium-1.40.1\dist\f01ded8a21fac2f76063c63489e9cd33968c52eb\language_server_windows_x64.exe --api_server_url https://server.codeium.com --manager_dir C:\Users\tambe\AppData\Local\Temp\89ae3cbf-930a-4bff-a36e-d380d0ae6cb5\codeium\manager --enable_chat_web_server --enable_lsp --ide_name vscode --inference_api_server_url https://inference.codeium.com --database_dir C:\Users\tambe\.codeium\database\9c0694567290725d9dcba14ade58e297 --enable_index_service --enable_local_search --search_max_workspace_file_count 5000 --indexed_files_retention_period_days 30 --ls_random_port_timeout_seconds 15 --workspace_id file_e_3A_TDS_Project_2_v0
    0        6   24768         C:\WINDOWS\system32\conhost.exe 0x4
    0      289   29900         c:\Users\tambe\.vscode\extensions\codeium.codeium-1.40.1\dist\f01ded8a21fac2f76063c63489e9cd33968c52eb\language_server_windows_x64.exe --api_server_url https://server.codeium.com --manager_dir C:\Users\tambe\AppData\Local\Temp\89ae3cbf-930a-4bff-a36e-d380d0ae6cb5\codeium\manager --enable_chat_web_server --enable_lsp --ide_name vscode --inference_api_server_url https://inference.codeium.com --database_dir C:\Users\tambe\.codeium\database\9c0694567290725d9dcba14ade58e297 --enable_index_service --enable_local_search --search_max_workspace_file_count 5000 --indexed_files_retention_period_days 30 --ls_random_port_timeout_seconds 15 --workspace_id file_e_3A_TDS_Project_2_v0 --run_child --limit_go_max_procs 4 --random_port --random_port_dir=C:\Users\tambe\AppData\Local\Temp\89ae3cbf-930a-4bff-a36e-d380d0ae6cb5\codeium\manager/child_random_port_1742454888954404100_4382656401991098810 --manager_lock_file=C:\Users\tambe\AppData\Local\Temp\89ae3cbf-930a-4bff-a36e-d380d0ae6cb5\codeium\manager/locks/manager.lock --child_lock_file C:\Users\tambe\AppData\Local\Temp\89ae3cbf-930a-4bff-a36e-d380d0ae6cb5\codeium\manager/locks/child_lock_1742454888955540000_766690306126769263
    0       41   26020       electron-nodejs (serverMain.js )
    0       37   28644       "C:\Users\tambe\AppData\Local\Programs\Microsoft VS Code\Code.exe" "c:\Users\tambe\AppData\Local\Programs\Microsoft VS Code\resources\app\extensions\json-language-features\server\dist\node\jsonServerMain" --node-ipc --clientProcessId=1712
    0       66   28924       electron-nodejs (server.js )
    0       44   29944       c:\Users\tambe\AppData\Local\Microsoft\WindowsApps\python3.12.exe c:\Users\tambe\.vscode\extensions\ms-python.autopep8-2025.2.0\bundled\tool\lsp_server.py
    0        6   24924         C:\WINDOWS\system32\conhost.exe 0x4
    0       92   13012  shared-process
    0      113   13496     gpu-process
    0      219   14952  window [1] (ga_ques.txt - Project_2_v0 - Visual Studio Code)
    0       46   21592  fileWatcher [1]
    0       31   24748     utility-network-service
    0       81   28996  ptyHost
    0       76    1596       C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe -noexit -command "try { . \"c:\Users\tambe\AppData\Local\Programs\Microsoft VS Code\resources\app\out\vs\workbench\contrib\terminal\common\scripts\shellIntegration.ps1\" } catch {}"
    0        8   22112       conpty-agent

Workspace Stats:
|  Window (ga_ques.txt - Project_2_v0 - Visual Studio Code)
|    Folder (Project_2_v0): 4 files
|      File types: py(2) txt(2)
|      Conf files:
    """


def ga_1_2(input_text: str):
    """
    Extracts the parameter value for 'email' from the input text,
    constructs a URL-encoded HTTPS request, and returns the JSON response.

    Args:
        input_text (str): The input text containing the task description.

    Returns:
        dict: The JSON response from the HTTP request.
    """
    # Extract the email parameter using regex
    match = re.search(r'email set to ([\w.@+-]+)', input_text)
    if not match:
        raise ValueError("Email parameter not found in the input text")

    email_param = match.group(1)
    url = "https://httpbin.org/get"
    params = {"email": email_param}

    # Send HTTPS GET request
    response = requests.get(url, params=params)
    return response.json()


def ga_1_3(file_path: str):
    """
    Runs `npx -y prettier@3.4.2 <file_path> | sha256sum` and returns the hash.

    :param file_path: The path to the file to format and hash.
    :return: The SHA-256 hash of the formatted file.
    """
    try:
        # Run Prettier and hash command
        result = subprocess.run(
            f"npx -y prettier {file_path} | sha256sum",
            shell=True,
            capture_output=True,
            text=True
        )

        # Check for errors
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"

        # Extract and return the hash value
        return result.stdout.strip().split()[0]

    except Exception as e:
        return f"Exception occurred: {str(e)}"
    finally:
        # Cleanup: Delete the temp file after processing
        if os.path.exists(file_path):
            os.remove(file_path)


def ga_1_4(question: str):
    """
    Extracts and evaluates a Google Sheets formula from a given question.

    :param question: A string containing a Google Sheets formula in the format:
                     "Let's make sure you can write formulas in Google Sheets. ... =SUM(...) What is the result?"
    :return: The computed numeric result of the extracted formula.
    """
    # Extract formula from the question
    formula_match = re.search(r'=(SUM\(.*\))', question)
    if not formula_match:
        raise ValueError("Invalid question: No valid formula found")

    formula = formula_match.group(1)

    # Extract SEQUENCE parameters
    seq_pattern = r"SEQUENCE\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)"
    seq_match = re.search(seq_pattern, formula)
    if not seq_match:
        raise ValueError("Invalid formula: SEQUENCE parameters not found")

    rows, cols, start, step = map(int, seq_match.groups())

    # Generate the sequence matrix
    sequence_matrix = np.arange(
        start, start + (rows * cols * step), step).reshape(rows, cols)

    # Extract ARRAY_CONSTRAIN parameters
    constrain_pattern = r"ARRAY_CONSTRAIN\(SEQUENCE\([^\)]+\),\s*(\d+),\s*(\d+)\)"
    constrain_match = re.search(constrain_pattern, formula)
    if not constrain_match:
        raise ValueError(
            "Invalid formula: ARRAY_CONSTRAIN parameters not found")

    constrain_rows, constrain_cols = map(int, constrain_match.groups())

    # Apply constraints (extract required portion)
    constrained_array = sequence_matrix[:constrain_rows, :constrain_cols]

    # Compute SUM
    return np.sum(constrained_array)


def ga_1_5(question: str):
    """
    Extracts and evaluates an Excel formula of the form:
    =SUM(TAKE(SORTBY({array}, {sort_order}), rows, cols))

    :param question: The question containing the Excel formula.
    :return: Computed numeric result.
    """

    # Extract the formula from the question (Handles { } instead of [ ])
    formula_match = re.search(
        r"=SUM\(\s*TAKE\(\s*SORTBY\(\s*\{([\d,\s]+)\}\s*,\s*\{([\d,\s]+)\}\s*\)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*\)",
        question,
        re.DOTALL
    )

    if not formula_match:
        print("❌ Regex did NOT match the formula!")  # Debugging
        raise ValueError("Invalid question format: Formula not found")

    # Extract parts from regex
    values_str = formula_match.group(1)
    sort_order_str = formula_match.group(2)
    take_rows = int(formula_match.group(3))
    take_cols = int(formula_match.group(4))

    # print("✅ Extracted Values:", values_str)
    # print("✅ Extracted Sort Order:", sort_order_str)
    # print("✅ Extracted Rows:", take_rows)
    # print("✅ Extracted Cols:", take_cols)

    values = list(map(int, values_str.split(",")))
    sort_order = list(map(int, sort_order_str.split(",")))

    if len(values) != len(sort_order):
        raise ValueError(
            "SORTBY error: Array and sort order lengths do not match")

    # Sorting values based on sort_order
    sorted_indices = np.argsort(sort_order)
    sorted_values = np.array(values)[sorted_indices]

    # Apply TAKE operation (Extract first `take_cols` elements)
    taken_values = sorted_values[:take_cols]

    # Compute SUM
    result = np.sum(taken_values)

    # print("✅ Final Computed Sum:", result)
    return result


def ga_1_6(file_path: str):
    """
    Extracts the value of a hidden input field from an HTML file.

    :param file_path: Path to the HTML file.
    :return: The value of the hidden input field if found, else None.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        pattern = r'<input\s+type=["\']hidden["\']\s+value=["\']([^"\']+)["\']'
        match = re.search(pattern, html_content, re.IGNORECASE)
        return match.group(1) if match else None

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


def ga_1_7(question: str) -> int:
    """
    Extracts the start date, end date, and weekday from a question and
    counts how many times that weekday appears in the given date range.

    :param question: A natural language question containing a date range and a weekday.
    :return: The count of the specified weekday in the given date range.
    """

    # Updated regex pattern to match different question formats
    pattern = r"(?:How many|Count the) (\w+)s? (?:are there in|between) (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})"

    match = re.search(pattern, question, re.IGNORECASE)

    if not match:
        raise ValueError(
            "Invalid question format. Could not extract date range or weekday.")

    weekday, start_date, end_date = match.groups()

    # Convert plural weekday (e.g., "Wednesdays") to singular form ("Wednesday")
    weekday = weekday.rstrip('s').capitalize()

    # Validate weekday
    weekdays = ["Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"]

    if weekday not in weekdays:
        raise ValueError(
            f"Invalid weekday: {weekday}. Expected one of {weekdays}")

    target_weekday = weekdays.index(weekday)

    # Convert strings to datetime objects
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Count occurrences of the specified weekday in the date range (including endpoints)
    count = sum(1 for i in range((end - start).days + 1)
                if (start + timedelta(days=i)).weekday() == target_weekday)

    return count


def ga_1_8(question: str, zip_file_path: str) -> list:
    """
    Extracts the specified column from a CSV file inside a ZIP archive.

    :param question: The question containing the ZIP filename, CSV filename, and column name.
    :param zip_file_path: Path to the ZIP file.
    :return: List of values in the specified column.
    """
    # Strict regex pattern to extract ZIP filename, CSV filename, and column name inside double quotes
    pattern = r'Download and unzip file ([\w\-.]+).*?\b([\w\-.]+\.csv)\b.*?(?:value in the|column) ["\'](.*?)["\']'

    match = re.search(pattern, question, re.IGNORECASE)

    if not match:
        raise ValueError(
            "Could not extract ZIP filename, CSV filename, or column name from the question.")

    zip_filename, csv_filename, column_name = match.groups()
    column_name = column_name.strip()

    # Validate ZIP file name
    if zip_filename != os.path.basename(zip_file_path):
        raise ValueError(
            f"Provided ZIP file '{os.path.basename(zip_file_path)}' does not match expected '{zip_filename}'.")

    extracted_values = []

    # Open and extract CSV from ZIP
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        if csv_filename not in z.namelist():
            raise FileNotFoundError(
                f"{csv_filename} not found in the ZIP archive.")

        with z.open(csv_filename) as f:
            reader = csv.DictReader(f.read().decode('utf-8').splitlines())

            # Validate extracted column name
            if column_name not in reader.fieldnames:
                raise ValueError(
                    f"Column '{column_name}' not found in CSV file. Available columns: {reader.fieldnames}")

            extracted_values = [row[column_name]
                                for row in reader if column_name in row]

    return extracted_values[0]


def ga_1_9(task_description: str) -> str:
    """
    Extracts JSON data from a task description, sorts the JSON based on available fields,
    and returns the result in a compact JSON format.

    :param task_description: A string containing the task description with JSON data.
    :return: A JSON string in the format {"answer": "<sorted_json>"} with compact spacing.
    """
    # Extract JSON data from the task description
    json_pattern = r'(\[.*\])'  # Matches JSON array inside square brackets
    json_match = re.search(json_pattern, task_description, re.DOTALL)

    if not json_match:
        raise ValueError(
            "Could not extract JSON data from the task description.")

    json_data = json_match.group(1).strip()

    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        raise ValueError("Extracted JSON data is not valid.")

    if not data or not isinstance(data, list) or not isinstance(data[0], dict):
        raise ValueError("JSON data should be a list of objects.")

    # Sort by 'age' first, then by 'name'
    sorted_data = sorted(data, key=lambda x: (x["age"], x["name"]))

    # Convert to compact JSON
    sorted_json = json.dumps(sorted_data, separators=(',', ':'))

    # Return in required format
    return json.dumps({"answer": sorted_json}, separators=(',', ':'))


def ga_1_10(file_path: str) -> str:
    """
    Converts a key=value formatted text file into a JSON object and returns its SHA-256 hash.

    :param file_path: Path to the input text file containing key=value pairs.
    :param pair_format: The delimiter separating keys and values (default: "=").
    :return: SHA-256 hash of the JSON object.
    """
    json_obj = {}
    pair_format = "="
    # Read file and process lines
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line and pair_format in line:
                key, value = line.split(pair_format, 1)
                json_obj[key.strip()] = value.strip()

    # Convert dictionary to compact JSON
    json_str = json.dumps(json_obj, separators=(',', ':'))
    # print(json_str)  # Debugging: Print the JSON string
    # Compute SHA-256 hash
    hash_hex = hashlib.sha256(json_str.encode()).hexdigest()

    return hash_hex


def ga_1_11(file_path: str, ) -> int:
    """
    Parses an HTML file and sums the 'data-value' attributes of all elements with a given tag and class.

    :param file_path: Path to the HTML file.
    :param tag: The HTML tag to search for (e.g., 'div', 'a').
    :param class_name: The class name to filter elements by.
    :return: Sum of 'data-value' attributes.
    """
    tag = "div"
    class_name = "foo"
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Find all elements with the specified tag and class
    elements = soup.find_all(tag, class_=class_name)

    # Sum up 'data-value' attributes
    total = sum(int(el["data-value"])
                for el in elements if el.has_attr("data-value"))

    return total


def ga_1_12(task_description: str, zip_path: str) -> int:
    """
    Parses a task description to extract file specifications (filenames and encodings),
    extracts and processes CSV/TSV files from a ZIP archive, and sums values where the first 
    column matches any extracted symbol.

    :param task_description: Task description containing file details and symbols.
    :param zip_path: Path to the ZIP file.
    :return: Sum of all values where the first column matches any extracted symbol.
    """
    # Extract file names and their encodings
    file_specs = {}
    file_pattern = re.findall(
        r'(\S+\.(csv|txt)):\s.*?encoded\s+in\s+([\w-]+)', task_description, re.IGNORECASE)
    for file_name, _, encoding in file_pattern:
        # Normalize encoding names for compatibility
        encoding = encoding.lower()
        if encoding == "cp-1252":
            encoding = "latin1"  # Use "latin1" instead of "CP-1252" for Linux/WSL
        file_specs[file_name] = encoding

    # Extract symbols from the task and normalize them to Unicode
    symbol_pattern = re.search(r'matches\s+(.*?)\s+across', task_description)
    if not symbol_pattern:
        raise ValueError(
            "Could not extract symbols from the task description.")

    symbols = {unicodedata.normalize('NFKC', s.strip())
               for s in symbol_pattern.group(1).split(" OR ")}

    total_sum = 0
    extract_dir = "extracted_files"

    # Extract ZIP contents
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Process each file with its respective encoding
    for filename, encoding in file_specs.items():
        file_path = os.path.join(extract_dir, filename)
        delimiter = "\t" if filename.endswith(".txt") else ","

        try:
            df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
            df.columns = df.columns.str.strip()  # Remove unexpected spaces

            # Ensure the file has exactly 2 columns
            if df.shape[1] != 2:
                raise ValueError(f"Unexpected number of columns in {filename}")

            df.columns = ["symbol", "value"]  # Standardize column names

            # Convert 'symbol' to Unicode and normalize
            df["symbol"] = df["symbol"].astype(str).map(
                lambda x: unicodedata.normalize('NFKC', x.strip()))
            df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

            # Debugging: Print unmatched symbols
            unmatched = df[~df["symbol"].isin(symbols)]["symbol"].unique()
            if unmatched.any():
                print(f"Unmatched symbols in {filename}: {unmatched}")

            # Filter rows by normalized symbols and sum the values
            total_sum += df[df["symbol"].isin(symbols)]["value"].sum()

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return int(total_sum)  # Return as integer


def ga_1_13(task_description: str) -> str:
    """
    Automates GitHub repository creation and file upload using GitHub API.

    :param task_description: Task description containing the required email for email.json.
    :param github_token: GitHub personal access token with 'repo' scope.
    :return: The raw GitHub URL of the uploaded email.json file.
    """
    # Extract email using regex
    match = re.search(r'"email":\s*"([^"]+)"', task_description)
    if not match:
        raise ValueError("Email address not found in task description.")

    email_value = match.group(1)

    # GitHub API credentials
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Step 1: Get authenticated user details
    user_response = requests.get(
        "https://api.github.com/user", headers=headers)
    if user_response.status_code != 200:
        raise ValueError(
            f"Failed to authenticate with GitHub: {user_response.text}")

    github_id = user_response.json()["login"]

    # Step 2: Create a new public repository
    repo_name = "email-json-repo"
    repo_url = "https://api.github.com/user/repos"

    repo_payload = {"name": repo_name, "private": False}
    repo_response = requests.post(repo_url, headers=headers, json=repo_payload)

    # 422 if repo already exists
    if repo_response.status_code not in [201, 422]:
        raise ValueError(f"Failed to create repository: {repo_response.text}")

    # Step 3: Create email.json content
    file_content = json.dumps({"email": email_value}, indent=4)

    # **Correct Encoding: Base64**
    encoded_content = base64.b64encode(
        file_content.encode("utf-8")).decode("utf-8")

    file_url = f"https://api.github.com/repos/{github_id}/{repo_name}/contents/email.json"

    file_payload = {
        "message": "Added email.json",
        "content": encoded_content,  # Base64 encoding
        "branch": "main"
    }

    file_response = requests.put(file_url, headers=headers, json=file_payload)

    if file_response.status_code not in [201, 200]:
        raise ValueError(f"Failed to upload file: {file_response.text}")

    # Step 4: Return raw file URL
    raw_url = f"https://raw.githubusercontent.com/{github_id}/{repo_name}/main/email.json"
    return raw_url


def ga_1_14(zip_path: str) -> str:
    """
    Unzips the given ZIP file, replaces occurrences of a text string (case-insensitive) in all files,
    and calculates the SHA-256 hash of the concatenated file contents.

    :param zip_path: Path to the ZIP file.
    :return: SHA-256 hash string of processed files.
    """
    # Safe temporary directory for extraction
    extract_dir = tempfile.mkdtemp()

    search_text = "IITM"
    replace_text = "IIT Madras"

    # Extract ZIP contents
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    except Exception as e:
        raise ValueError(f"Error extracting ZIP file: {e}")

    # Regex pattern for case-insensitive replacement
    pattern = re.compile(re.escape(search_text), re.IGNORECASE)

    # Process each file in extracted directory
    for root, _, files in os.walk(extract_dir):
        for filename in sorted(files):  # Sorting to match "cat *" order
            file_path = os.path.join(root, filename)

            # Ensure file is readable and writable
            try:
                os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
            except Exception as e:
                print(
                    f"⚠ Warning: Could not change permissions for {file_path}: {e}")

            try:
                # Read file in binary mode to avoid encoding issues
                with open(file_path, "rb") as file:
                    content = file.read()

                # Replace occurrences in text mode
                decoded_content = content.decode("utf-8", errors="ignore")
                new_content = pattern.sub(replace_text, decoded_content)

                # Write back only if changes were made
                if new_content != decoded_content:
                    with open(file_path, "wb") as file:
                        file.write(new_content.encode("utf-8"))

            except Exception as e:
                print(f"⚠ Warning: Skipping {file_path} due to error: {e}")

    # Compute SHA-256 hash after processing all files
    sha256_hash = hashlib.sha256()

    # Collect all files (sorted) for hashing
    all_files = []
    for root, _, files in os.walk(extract_dir):
        for filename in sorted(files):
            all_files.append(os.path.join(root, filename))

    # Hash files in sorted order
    for file_path in sorted(all_files):
        try:
            with open(file_path, "rb") as file:
                # Read in chunks to handle large files
                while chunk := file.read(8192):
                    sha256_hash.update(chunk)
        except Exception as e:
            print(
                f"⚠ Warning: Skipping {file_path} in hashing due to error: {e}")

    return sha256_hash.hexdigest()


def ga_1_15(task_description: str, zip_path: str) -> int:
    """
    Extracts parameters from a task description, processes a ZIP file, 
    and calculates the total size of files meeting the conditions.

    :param task_description: Task description containing the file size and date.
    :param zip_path: Path to the ZIP file.
    :return: Total size of all matching files.
    """

    # ✅ Extract minimum file size (in bytes)
    size_match = re.search(r'at\s+least\s+(\d+)\s+bytes', task_description)
    if not size_match:
        raise ValueError("Minimum file size not found in task description.")
    min_size = int(size_match.group(1))

    # ✅ Extract minimum modification date (Fix regex to avoid extra text)
    date_match = re.search(
        r'on or after\s+([A-Za-z]{3},\s\d{1,2}\s[A-Za-z]{3},\s\d{4},\s\d{1,2}:\d{2}\s[ap]m\sIST)',
        task_description,
    )
    if not date_match:
        raise ValueError(
            "Minimum modification date not found in task description.")

    min_date_str = date_match.group(1)  # ✅ Extract only the date part

    # ✅ Convert date string to datetime object
    min_date = datetime.strptime(min_date_str, "%a, %d %b, %Y, %I:%M %p IST")

    # ✅ Use a temporary directory to avoid permission issues
    extract_dir = tempfile.mkdtemp()

    # ✅ Extract ZIP contents while preserving timestamps
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    total_size = 0

    # ✅ Process extracted files
    for root, _, files in os.walk(extract_dir):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Get file size
            file_size = os.path.getsize(file_path)

            # Get file modification time
            mod_time = os.path.getmtime(file_path)
            mod_date = datetime.fromtimestamp(
                mod_time)  # ✅ FIXED datetime issue

            # ✅ Check conditions: size and modification date
            if file_size >= min_size and mod_date >= min_date:
                total_size += file_size

    return total_size


def ga_1_16(zip_path: str) -> str:
    """
    Extracts a ZIP file, moves all files into a single directory, renames them by shifting digits,
    and computes the SHA-256 hash of sorted file contents.

    :param zip_path: Path to the ZIP file.
    :return: SHA-256 hash of sorted file contents.
    """
    extract_dir = "extracted_files"
    merged_dir = "merged_files"

    # ✅ Step 1: Extract ZIP contents
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # ✅ Step 2: Move all files to `merged_files`, ensuring uniqueness
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    os.makedirs(merged_dir, exist_ok=True)

    existing_files = set()

    for root, _, files in os.walk(extract_dir):
        for file in files:
            old_path = os.path.join(root, file)

            # Ensure unique filename when moving
            new_filename = file
            counter = 1
            while new_filename in existing_files:
                name, ext = os.path.splitext(file)
                new_filename = f"{name}_{counter}{ext}"
                counter += 1

            new_path = os.path.join(merged_dir, new_filename)
            shutil.move(old_path, new_path)
            existing_files.add(new_filename)

    # ✅ Step 3: Rename files by shifting digits
    def shift_digits(filename):
        return re.sub(r'\d', lambda x: str((int(x.group()) + 1) % 10), filename)

    final_files = {}
    for filename in sorted(os.listdir(merged_dir)):  # Sort to ensure consistency
        old_path = os.path.join(merged_dir, filename)
        new_filename = shift_digits(filename)

        # Ensure unique filenames after renaming
        counter = 1
        final_name = new_filename
        while final_name in final_files:
            name, ext = os.path.splitext(new_filename)
            final_name = f"{name}_{counter}{ext}"
            counter += 1

        new_path = os.path.join(merged_dir, final_name)
        os.rename(old_path, new_path)
        final_files[final_name] = new_path

    # ✅ Step 4: Compute SHA-256 hash of sorted file contents
    all_contents = []

    for filename in sorted(final_files.keys()):  # Sort files for consistency
        file_path = final_files[filename]
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                all_contents.append(line.strip())

    # ✅ Step 5: Sort contents like `LC_ALL=C sort`
    all_contents.sort()

    # ✅ Step 6: Compute SHA-256 hash
    sha256_hash = hashlib.sha256(
        "\n".join(all_contents).encode("utf-8")).hexdigest()
    return sha256_hash


def ga_1_17(zip_path: str, ) -> int:
    """
    Extracts the given ZIP file and compares two specified files line by line to count differing lines.

    :param zip_path: Path to the ZIP file.
    :param file1: First file to compare.
    :param file2: Second file to compare.
    :return: Number of differing lines.
    """
    extract_dir = "extracted_files"
    file1, file2 = "a.txt", "b.txt"
    # Extract ZIP contents
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    file1_path = os.path.join(extract_dir, file1)
    file2_path = os.path.join(extract_dir, file2)

    # Read both files and compare line by line
    with open(file1_path, "r", encoding="utf-8", errors="ignore") as f1, \
            open(file2_path, "r", encoding="utf-8", errors="ignore") as f2:
        diff_count = sum(1 for line1, line2 in zip(f1, f2)
                         if line1.strip() != line2.strip())

    return diff_count


def ga_1_18():
    query = f"""\
SELECT SUM(units * price)
FROM tickets
WHERE TRIM(LOWER(type)) LIKE 'gold';
    """
    return query

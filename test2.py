import json
import os
import pickle
import subprocess

import requests

# Define the API endpoint
# Change this if your server is running elsewhere
API_URL = "http://localhost:8000/api"

# Sample test cases
test_cases = [
    # {
    #     "question": "What is the output of code -s?",
    #     "expected_answer": "ga_1_1",
    # "file": ""
    # },
    # {
    #     "question": "Running uv run --with httpie -- https [URL] sends a HTTPS request to the URL.\nSend a HTTPS request to https://httpbin.org/get with the URL encoded parameter email set to 22f3002248@ds.study.iitm.ac.in\nWhat is the JSON output of the command? (Paste only the JSON body, not the headers)",
    #     "expected_answer": "ga_1_2",
    # "file": ""
    # },
    # {
    #     "question": "run npx -y prettier@3.4.2 README.md | sha256sum.\nWhat is the output of the command?",
    #     "expected_answer": "ga_1_3",
    #     "file": "data/README.md"
    # },
    # {
    #     "question": "Type this formula into Google Sheets. (It won't work in Excel)\n=SUM(ARRAY_CONSTRAIN(SEQUENCE(100, 100, 15, 7), 1, 10))\nWhat is the result?",
    #     "expected_answer": "ga_1_4",
    #     "file": ""
    # },
    # {
    #     "question": "Type this formula into Excel.\nNote: This will ONLY work in Office 365.\n=SUM(TAKE(SORTBY({5,9,3,2,14,4,3,1,1,13,2,12,12,11,12,6}, {10,9,13,2,11,8,16,14,7,15,5,4,6,1,3,12}), 1, 9))\nWhat is the result?\nNote: If you get #NAME? you have the wrong version of Excel. Find a friend for whom this works.",
    #     "expected_answer": "ga_1_5",
    #     "file": ""
    # },
    # {
    #     "question": "Find the hidden input field's value in the given HTML snippet. What is the extracted value?",
    #     "expected_answer": "ga_1_6",
    #     "file": "data/q6.html"  # ! NEEDS CHECKING
    # },
    # {
    #     "question": "Count the Wednesdays between 1983-06-08 to 2013-05-08, including both endpoints.",
    #     "expected_answer": "ga_1_7",
    #     "file": ""
    # },
    {
        "question": "Extract extract.csv from q-extract-csv-zip.zip and find the value in the 'answer' column.",
        "expected_answer": "ga_1_8",
        "file": "data/q-extract-csv-zip.zip"
    },
    {
        "question": "Sort the given JSON by age, then by name. Return the sorted JSON as a single line. [{\"name\":\"Alice\",\"age\":80},{\"name\":\"Bob\",\"age\":52},{\"name\":\"Charlie\",\"age\":1},{\"name\":\"David\",\"age\":10},{\"name\":\"Emma\",\"age\":34},{\"name\":\"Frank\",\"age\":17},{\"name\":\"Grace\",\"age\":79},{\"name\":\"Henry\",\"age\":61},{\"name\":\"Ivy\",\"age\":66},{\"name\":\"Jack\",\"age\":25},{\"name\":\"Karen\",\"age\":21},{\"name\":\"Liam\",\"age\":69},{\"name\":\"Mary\",\"age\":78},{\"name\":\"Nora\",\"age\":25},{\"name\":\"Oscar\",\"age\":12},{\"name\":\"Paul\",\"age\":66}]",
        "expected_answer": "ga_1_9",
        "file": ""
    },
    # {
    #     "question": "Convert the contents of q-multi-cursor-json.txt into a valid JSON object. Compute its hash at tools-in-data-science.pages.dev/jsonhash. What is the hash?",
    #     "expected_answer": "ga_1_10",
    #     "file": "data/q-multi-cursor-json.txt"
    # },
    # {
    #     "question": "Find all <div> elements with class foo and sum their data-value attributes. What is the sum?",
    #     "expected_answer": "ga_1_11",
    #     "file": ""
    # },
    # {
    #     "question": "Extract and process q-unicode-data.zip to sum values of symbols matching † OR Š OR ….",
    #     "expected_answer": "ga_1_12",
    #     "file": "data/q-unicode-data.zip"
    # },
    # {
    #     "question": "Create a GitHub repository and push email.json with \"email\": \"random_email@example.com\". Provide the raw file URL.",
    #     "expected_answer": "ga_1_13",
    #     "file": ""
    # },
    # {
    #     "question": "Unzip q-replace-across-files.zip, replace all occurrences of IITM with IIT Madras, and compute the sha256sum using cat * | sha256sum. What is the result?",
    #     "expected_answer": "ga_1_14",
    #     "file": "data/q-replace-across-files.zip"
    # },
    # {
    #     "question": "Extract q-list-files-attributes.zip and list all files with sizes and dates. Find the total size of files >= 10000 bytes and modified after Wed, 15 Aug, 2012, 5:45 pm IST.",
    #     "expected_answer": "ga_1_15",
    #     "file": "data/q-list-files-attributes.zip"
    # },
    # {
    #     "question": "Extract q-move-rename-files.zip, move all files into a single folder, rename digits, and compute grep . * | LC_ALL=C sort | sha256sum. What is the result?",
    #     "expected_answer": "ga_1_16",
    #     "file": "data/q-move-rename-files.zip"
    # },
    # {
    #     "question": "Extract q-compress-files.zip and compare a.txt and b.txt. How many lines differ?",
    #     "expected_answer": "ga_1_17",
    #     "file": "data/q-compress-files.zip"
    # },
    # {
    #     "question": "In a SQLite tickets table, find the total sales for 'Gold' tickets (case-insensitive) as SUM(Units * Price). What is the sum?",
    #     "expected_answer": "ga_1_18",
    #     "file": ""
    # }
]

# Run test cases

# count = 0
# for i, test in enumerate(test_cases, 1):
#     response = requests.post(API_URL, data={"question": test["question"]},
#                              files={"file": test["file"]})

#     print(f"Test Case {i}:")
#     print("Input Question:", test["question"])
#     print("-" * 80)
#     if response.status_code == 200:
#         ans = response.json()  # ✅ Use this instead of json.loads(response)
#         print({"answer": ans["answer"]})

#     print("-" * 80)
# # print(f"✅ Passed Test Cases: {count} / {len(test_cases)} ✅")
# # print("-" * 80)

for i, test in enumerate(test_cases, 1):
    question = test["question"]
    file_path = test["file"]

    # Construct the curl command
    if file_path:
        absolute_path = os.path.abspath(file_path)
        curl_command = f'curl -X POST "{API_URL}" -F "question={question}" -F "file=@{absolute_path}"'
    else:
        curl_command = f'curl -X POST "{API_URL}" -F "question={question}"'

    # Print and execute the command
    print(f"Test Case {i}:")
    print("Input Question:", question)
    print("Executing:", curl_command)
    print("-" * 80)

    result = subprocess.run(curl_command, shell=True,
                            capture_output=True, text=True)

    # Print response
    print("Response:")
    print(result.stdout)
    print("-" * 80)

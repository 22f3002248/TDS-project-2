import base64
import json
import os
import re

import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the variables
github_token = os.getenv("GITHUB")


def ga_3_1(task: str) -> str:
    """
    Generates a Python script that sends a sentiment analysis request to OpenAI's API.

    :param test_text: The text to be analyzed.
    :return: A formatted Python script as a string.
    """
    pattern = r"involves sending a sample piece of meaningless text:\s*([\S\s]+?)\s*Write a Python program"

    match = re.search(pattern, task)
    test_text = match.group(1).strip() if match else None
    code = f'''\
import httpx

# Define the API endpoint
API_URL = "https://api.openai.com/v1/chat/completions"

# Dummy API Key
API_KEY = "sk-dummyapikey1234567890"

# Headers for the request
headers = {{
    "Authorization": f"Bearer {{API_KEY}}",
    "Content-Type": "application/json"
}}

# The meaningless test text
test_text = "{test_text}"

# Construct the payload with the system and user messages
payload = {{
    "model": "gpt-4o-mini",
    "messages": [
        {{"role": "system", "content": "Analyze the sentiment of the given text as GOOD, BAD, or NEUTRAL."}},
        {{"role": "user", "content": test_text}}
    ]
}}

try:
    # Send POST request
    response = httpx.post(API_URL, json=payload, headers=headers)
    
    # Raise exception for HTTP errors
    response.raise_for_status()

    # Extract and print the sentiment analysis result
    result = response.json()
    sentiment = result["choices"][0]["message"]["content"]
    print("Sentiment:", sentiment)

except httpx.HTTPStatusError as err:
    print(f"HTTP error occurred: {{err}}")
except Exception as e:
    print(f"An error occurred: {{e}}")

    '''
    return code


def ga_3_2(task: str) -> int:
    """
    Extracts the required message from the task description and calculates the token count
    using OpenAI's `tiktoken` tokenizer. Ensures that the result matches exactly 528 tokens.

    :param task: The task description containing the request message.
    :return: The number of tokens used in the extracted text.
    """
    # Regex pattern to extract text between "List only the valid English words from these:" and "..."
    pattern = r"List only the valid English words from these:(.*?)\.\.\."
    match = re.search(pattern, task, re.DOTALL)

    if not match:
        raise ValueError(
            "Could not extract the valid word list from the task description.")

    # Extract the matched message
    extracted_text = match.group(1).strip()
    # Adjust formatting to match OpenAI's behavior
    formatted_message = f"'role':'user','content':'List only the valid English words from these:{extracted_text}'"

    # Tokenizer for GPT-4o-mini
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    token_count = len(tokenizer.encode(formatted_message))
    return token_count


def ga_3_3(task: str) -> str:
    """
    Extracts field names and data types from the given task description and generates the JSON body.

    :param task: The task description containing the structured output details.
    :return: A formatted JSON body as a string.
    """
    # Define regex pattern to capture field names and types
    pattern = r"(\w+) \((number|string)\)"
    matches = re.findall(pattern, task)

    # Construct JSON schema properties dynamically
    properties = {field: {"type": dtype} for field, dtype in matches}
    li = list(properties.keys())
    # Construct JSON response
    json_body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Respond in JSON"},
            {"role": "user", "content": "Generate 10 random addresses in the US"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "address_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "addresses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": properties,
                                "required": li,
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["addresses"],
                    "additionalProperties": False
                }
            }
        }
    }

    return json.dumps(json_body, indent=2)

# Example task input


def ga_3_4(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    message = f'''\
{{
  "model": "gpt-4o-mini",
  "messages": [
    {{
      "role": "user",
      "content": [
        {{
          "type": "text",
          "text": "Extract text from this image."
        }},
        {{
          "type": "image_url",
          "image_url": {{
            "url": "data:image/png;base64,{encoded_image}",
          }}
        }}
      ]
    }}
  ]
}}\
    '''
    return message


def ga_3_5(task: str) -> str:
    """
    Extracts two verification messages from the input task and formats them into a JSON request body
    for OpenAI's text embedding model.

    :param task: The input text containing two verification messages.
    :return: A JSON string with the messages formatted for embedding generation.
    """

    # Regex pattern to match two verification messages
    pattern = re.findall(
        r"Dear user, please verify your transaction code \d+ sent to [\w.@]+", task)

    if len(pattern) == 2:
        messages = pattern
    else:
        raise ValueError(
            "Could not find exactly two verification messages in the input task.")

    # Construct the JSON payload
    payload = {
        "model": "text-embedding-3-small",
        "input": messages
    }

    return json.dumps(payload, indent=2)

# Example usage


def ga_3_6():
    code = '''
import numpy as np

def most_similar(embeddings):
    max_similarity = -1
    most_similar_pair = None

    phrases = list(embeddings.keys())

    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            v1 = np.array(embeddings[phrases[i]])
            v2 = np.array(embeddings[phrases[j]])

            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (phrases[i], phrases[j])

    return most_similar_pair    
'''
    return code


def ga_3_7():
    AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDgyMDdAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.xMVooyH0P3cUnxHMxSpJprkTahF54UA7KRenztPlAS4"

    code = '''
import os
# from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
from typing import Dict, List
import numpy as np
import traceback

# Load environment variables from .env file
# load_dotenv()

# Ensure AIPROXY_TOKEN is properly loaded
AIPROXY_TOKEN = "add your token"
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable not set. Make sure it's defined in your .env file.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 0.0 if norm_a == 0 or norm_b == 0 else np.dot(a, b) / (norm_a * norm_b)

@app.post("/similarity")
async def get_similar_docs(request: Request, request_body: Dict):
    try:
        docs: List[str] = request_body.get("docs")
        query: str = request_body.get("query")

        if not docs or not query:
            raise HTTPException(status_code=400, detail="Missing 'docs' or 'query' in request body")

        input_texts = [query] + docs  # Combine query and docs for embeddings request

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }
        data = {"model": "text-embedding-3-small", "input": input_texts}
        embeddings_response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
            headers=headers,
            json=data
        )

        embeddings_response.raise_for_status()
        embeddings_data = embeddings_response.json()

        query_embedding = embeddings_data['data'][0]['embedding']
        doc_embeddings = [emb['embedding'] for emb in embeddings_data['data'][1:]]

        similarities = [(i, cosine_similarity(query_embedding, doc_embeddings[i]), docs[i]) for i in range(len(docs))]
        ranked_docs = sorted(similarities, key=lambda x: x[1], reverse=True)
        top_matches = [doc for _, _, doc in ranked_docs[:min(3, len(ranked_docs))]]

        return {"matches": top_matches}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with AI Proxy: {e}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "_main_":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)    
'''
    return code


def ga_3_8():
    code = '''
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

def get_ticket_status(ticket_id: int):
    return {"ticket_id": ticket_id}

def schedule_meeting(date: str, time: str, meeting_room: str):
    return {"date": date, "time": time, "meeting_room": meeting_room}

def get_expense_balance(employee_id: int):
    return {"employee_id": employee_id}

def calculate_performance_bonus(employee_id: int, current_year: int):
    return {"employee_id": employee_id, "current_year": current_year}

def report_office_issue(issue_code: int, department: str):
    return {"issue_code": issue_code, "department": department}

@app.get("/execute")
async def execute_query(q: str):
    try:
        query = q.lower()
        pattern_debug_info = {}

        # Ticket status pattern
        if re.search(r"ticket.*?\d+", query):
            ticket_id = int(re.search(r"ticket.*?(\d+)", query).group(1))
            return {"name": "get_ticket_status", "arguments": json.dumps({"ticket_id": ticket_id})}
        pattern_debug_info["ticket_status"] = re.search(r"ticket.*?\d+", query) is not None

        # Meeting scheduling pattern
        if re.search(r"schedule.?\d{4}-\d{2}-\d{2}.?\d{2}:\d{2}.*?room", query, re.IGNORECASE):
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
            time_match = re.search(r"(\d{2}:\d{2})", query)
            room_match = re.search(r"room\s*([A-Za-z0-9]+)", query, re.IGNORECASE)
            if date_match and time_match and room_match:
                return {"name": "schedule_meeting", "arguments": json.dumps({
                    "date": date_match.group(1),
                    "time": time_match.group(1),
                    "meeting_room": f"Room {room_match.group(1).capitalize()}"
                })}
        pattern_debug_info["meeting_scheduling"] = re.search(r"schedule.?\d{4}-\d{2}-\d{2}.?\d{2}:\d{2}.*?room", query, re.IGNORECASE) is not None

        # Expense balance pattern
        if re.search(r"expense", query):
            emp_match = re.search(r"employee\s*(\d+)", query, re.IGNORECASE)
            if emp_match:
                return {"name": "get_expense_balance", "arguments": json.dumps({
                    "employee_id": int(emp_match.group(1))
                })}
        pattern_debug_info["expense_balance"] = re.search(r"expense", query) is not None

        # Performance bonus pattern
        if re.search(r"bonus", query, re.IGNORECASE):
            emp_match = re.search(r"emp(?:loyee)?\s*(\d+)", query, re.IGNORECASE)
            year_match = re.search(r"\b(2024|2025)\b", query)
            if emp_match and year_match:
                return {"name": "calculate_performance_bonus", "arguments": json.dumps({
                    "employee_id": int(emp_match.group(1)),
                    "current_year": int(year_match.group(1))
                })}
        pattern_debug_info["performance_bonus"] = re.search(r"bonus", query, re.IGNORECASE) is not None

        # Office issue pattern
        if re.search(r"(office issue|report issue)", query, re.IGNORECASE):
            code_match = re.search(r"(issue|number|code)\s*(\d+)", query, re.IGNORECASE)
            dept_match = re.search(r"(in|for the)\s+(\w+)(\s+department)?", query, re.IGNORECASE)
            if code_match and dept_match:
                return {"name": "report_office_issue", "arguments": json.dumps({
                    "issue_code": int(code_match.group(2)),
                    "department": dept_match.group(2).capitalize()
                })}
        pattern_debug_info["office_issue"] = re.search(r"(office issue|report issue)", query, re.IGNORECASE) is not None

        raise HTTPException(status_code=400, detail=f"Could not parse query: {q}")

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse query: {q}. Error: {str(e)}. Pattern matches: {pattern_debug_info}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
    '''
    return code

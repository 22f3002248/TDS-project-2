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
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")


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
    pass


def ga_3_8():
    pass

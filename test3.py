import json

import requests

# This is just an example server to see what we see.

url = "https://httpbin.org/post"

my_multiline_string_answer = """This is a multiline
string that spans
multiple lines    with    spaces 
and some newlines
and a tab	as well."""

response_to_send_to_tds_evaluator = {
    "answer": my_multiline_string_answer
}

# Send the JSON data
response = requests.post(url, json=response_to_send_to_tds_evaluator)

# Check the response
print(response.status_code)
print(json.dumps(response.json()))
print(response.text)

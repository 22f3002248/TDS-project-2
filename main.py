# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi", "uvicorn"]
# ///
# uv header script
import json
import os
import pickle
import re
import traceback
from typing import Dict, List, Optional

import aiofiles
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import (Body, FastAPI, File, Form, HTTPException, Query, Request,
                     UploadFile)
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from ga1 import (ga_1_1, ga_1_2, ga_1_3, ga_1_4, ga_1_5, ga_1_6, ga_1_7,
                 ga_1_8, ga_1_9, ga_1_10, ga_1_11, ga_1_12, ga_1_13, ga_1_14,
                 ga_1_15, ga_1_16, ga_1_17, ga_1_18)
from ga2 import (ga_2_1, ga_2_2, ga_2_3, ga_2_4, ga_2_5, ga_2_6, ga_2_7,
                 ga_2_8, ga_2_9, ga_2_10)
from ga3 import ga_3_1, ga_3_2, ga_3_3, ga_3_4, ga_3_5, ga_3_6, ga_3_7, ga_3_8
from ga4 import (ga_4_1, ga_4_2, ga_4_3, ga_4_4, ga_4_5, ga_4_6, ga_4_7,
                 ga_4_8, ga_4_9, ga_4_10)
from ga5 import (ga_5_1, ga_5_2, ga_5_3, ga_5_4, ga_5_5, ga_5_6, ga_5_7,
                 ga_5_8, ga_5_9, ga_5_10)
from matching import find_best_match

# Load environment variables from .env file
load_dotenv()

# Access the variables
github_token = os.getenv("GITHUB")
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# with open("stored_data_long.pkl", "rb") as f:
#     data = pickle.load(f)
#     stored_data = data["stored_data"]
#     stored_embeddings = data["stored_embeddings"]


def process_response(map_, file):
    print(map_["input_question"])
    print("-"*50)
    print(map_["best_answer"])
    print("-"*50)
    # GA 1
    if map_["best_answer"] == "ga_1_1":
        return ga_1_1()

    elif map_["best_answer"] == "ga_1_2":
        return ga_1_2(map_["input_question"])

    elif map_["best_answer"] == "ga_1_3":
        print(f"File: {file}")
        if file is None:
            return "File not found"
        return ga_1_3(file)

    elif map_["best_answer"] == "ga_1_4":
        return ga_1_4(map_["input_question"])

    elif map_["best_answer"] == "ga_1_5":
        return ga_1_5(map_["input_question"])

    elif map_["best_answer"] == "ga_1_6":
        print(f"File: {file}")
        if file is None:
            return "File not found"
        return ga_1_6(file)

    elif map_["best_answer"] == "ga_1_7":
        return ga_1_7(map_["input_question"])

    elif map_["best_answer"] == "ga_1_8":
        print(f"File: {file}")
        if file is None:
            return "File not found"
        return ga_1_8(map_["input_question"], file)

    elif map_["best_answer"] == "ga_1_9":
        return ga_1_9(map_["input_question"])

    elif map_["best_answer"] == "ga_1_10":
        print(f"File: {file}")
        if file is None:
            return "File not found"
        return ga_1_10(file)

    elif map_["best_answer"] == "ga_1_11":
        print(f"File: {file}")
        if file is None:
            return "File not found"
        return ga_1_11(file)

    elif map_["best_answer"] == "ga_1_12":
        print(f"File: {file}")
        if file is None:
            return "File not found"
        return ga_1_12(map_["input_question"], file)

    elif map_["best_answer"] == "ga_1_13":
        return ga_1_13(map_["input_question"])

    elif map_["best_answer"] == "ga_1_14":
        print(f"File: {file}")
        if file is None:
            return "File not found"
        return ga_1_14(file)

    elif map_["best_answer"] == "ga_1_15":
        print(f"File: {file}")
        if file is None:
            return "File not found"
        return ga_1_15(map_["input_question"], file)

    elif map_["best_answer"] == "ga_1_16":
        print(f"File: {file}")
        if file is None:
            return "File not found"
        return ga_1_16(file)

    elif map_["best_answer"] == "ga_1_17":
        print(f"File: {file}")
        if file is None:
            return "File not found"
        return ga_1_17(file)

    elif map_["best_answer"] == "ga_1_18":
        return ga_1_18()
    # GA 2
    elif map_["best_answer"] == "ga_2_1":
        return ga_2_1()

    elif map_["best_answer"] == "ga_2_2":
        print(f"File: {file}")
        if file is None:
            return "File not found"
        return ga_2_2(file)

    elif map_["best_answer"] == "ga_2_3":
        return ga_2_3(map_["input_question"])

    elif map_["best_answer"] == "ga_2_4":
        return ga_2_4(map_["input_question"])

    elif map_["best_answer"] == "ga_2_5":
        return ga_2_5(file, map_["input_question"])

    elif map_["best_answer"] == "ga_2_6":
        return ga_2_6(file)

    elif map_["best_answer"] == "ga_2_7":
        return ga_2_7(map_["input_question"])

    elif map_["best_answer"] == "ga_2_8":
        return ga_2_8()

    elif map_["best_answer"] == "ga_2_9":
        load_csv(file)
        return "34.93.189.78:8000/ga_2_9"

    elif map_["best_answer"] == "ga_2_10":
        return ga_2_10()
    # GA 3
    elif map_["best_answer"] == "ga_3_1":
        return ga_3_1(map_["input_question"])

    elif map_["best_answer"] == "ga_3_2":
        return ga_3_2(map_["input_question"])

    elif map_["best_answer"] == "ga_3_3":
        return ga_3_3(map_["input_question"])

    elif map_["best_answer"] == "ga_3_4":
        return ga_3_4(file)

    elif map_["best_answer"] == "ga_3_5":
        return ga_3_5(map_["input_question"])

    elif map_["best_answer"] == "ga_3_6":
        return ga_3_6()

    elif map_["best_answer"] == "ga_3_7":
        return "34.93.189.78:8000/similarity"

    elif map_["best_answer"] == "ga_3_8":
        return "34.93.189.78:8000/execute"
    # GA 4
    elif map_["best_answer"] == "ga_4_1":
        return ga_4_1(map_["input_question"])

    elif map_["best_answer"] == "ga_4_2":
        return ga_4_2(map_["input_question"])

    elif map_["best_answer"] == "ga_4_3":
        return ga_4_3()

    elif map_["best_answer"] == "ga_4_4":
        return ga_4_4(map_["input_question"])

    elif map_["best_answer"] == "ga_4_5":
        return ga_4_5(map_["input_question"])

    elif map_["best_answer"] == "ga_4_6":
        return ga_4_6(map_["input_question"])

    elif map_["best_answer"] == "ga_4_7":
        return ga_4_7(map_["input_question"])

    elif map_["best_answer"] == "ga_4_8":
        return ga_4_8(map_["input_question"])

    elif map_["best_answer"] == "ga_4_9":
        return ga_4_9(file, map_["input_question"])

    elif map_["best_answer"] == "ga_4_10":
        return ga_4_10(file)
    # GA 5
    elif map_["best_answer"] == "ga_5_1":
        return ga_5_1(map_["input_question"])
    elif map_["best_answer"] == "ga_5_2":
        return ga_5_2(map_["input_question"])
    elif map_["best_answer"] == "ga_5_3":
        return ga_5_3(map_["input_question"])
    elif map_["best_answer"] == "ga_5_4":
        return ga_5_4(map_["input_question"])
    elif map_["best_answer"] == "ga_5_5":
        return ga_5_5(map_["input_question"])
    elif map_["best_answer"] == "ga_5_6":
        return ga_5_6(map_["input_question"])
    elif map_["best_answer"] == "ga_5_7":
        return ga_5_7(map_["input_question"])
    elif map_["best_answer"] == "ga_5_8":
        return ga_5_8(map_["input_question"])
    elif map_["best_answer"] == "ga_5_9":
        return ga_5_9(map_["input_question"])
    elif map_["best_answer"] == "ga_5_10":
        return ga_5_10(map_["input_question"])
    else:
        print("-----------------------------------No answer found-----------------------------------")


TEMP_DIR = os.path.abspath("tempdata")  # Ensure absolute path
os.makedirs(TEMP_DIR, exist_ok=True)    # Ensure the directory exists


async def save_file(file: UploadFile) -> str:
    """Save an uploaded file, overwriting if it already exists."""
    file_path = os.path.join(TEMP_DIR, file.filename)  # Save directly

    try:
        async with aiofiles.open(file_path, "wb") as buffer:
            print(f"Writing file to: {file_path}")
            while chunk := await file.read(1024 * 1024):  # Read 1MB chunks
                await buffer.write(chunk)

        print(f"File saved successfully: {file_path}")
        return file_path  # Return the saved file path

    except Exception as e:
        print(f"Error saving file: {e}")
        return None


@app.post("/api")
async def process_question(
    request: Request,
    question: str = Form(...),  # <-- Use Form() for form-encoded data
    file: Optional[UploadFile] = File(None)
):
    print("Received question:", repr(question))  # Debugging: check newlines
    print("Received file:", file.filename if file else "No file")

    map_ = find_best_match(question)

    # Convert NumPy values to Python native types
    for key, value in map_.items():
        if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            map_[key] = value.item()

    file_path = None
    if file:
        file_path = await save_file(file)

    result = process_response(map_, file_path)
    if isinstance(result, (np.integer, np.floating)):
        result = result.item()

    response = {"answer": result}
    return response  # JSON response

# ==================================================================================


# ga 2 7
students_data = None


def load_csv(csv_file):
    df = pd.read_csv(csv_file, dtype={"studentId": int, "class": str})
    students_data = df.to_dict(orient="records")


@app.get("/ga_2_7")
def ga_2_7(class_param: list[str] = Query(None, alias="class")):
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


# ga 3 7
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
            raise HTTPException(
                status_code=400, detail="Missing 'docs' or 'query' in request body")

        # Combine query and docs for embeddings request
        input_texts = [query] + docs

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
        doc_embeddings = [emb['embedding']
                          for emb in embeddings_data['data'][1:]]

        similarities = [(i, cosine_similarity(
            query_embedding, doc_embeddings[i]), docs[i]) for i in range(len(docs))]
        ranked_docs = sorted(similarities, key=lambda x: x[1], reverse=True)
        top_matches = [doc for _, _,
                       doc in ranked_docs[:min(3, len(ranked_docs))]]

        return {"matches": top_matches}

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500, detail=f"Error communicating with AI Proxy: {e}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


# ga 3 8
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
        pattern_debug_info["ticket_status"] = re.search(
            r"ticket.*?\d+", query) is not None

        # Meeting scheduling pattern
        if re.search(r"schedule.?\d{4}-\d{2}-\d{2}.?\d{2}:\d{2}.*?room", query, re.IGNORECASE):
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
            time_match = re.search(r"(\d{2}:\d{2})", query)
            room_match = re.search(
                r"room\s*([A-Za-z0-9]+)", query, re.IGNORECASE)
            if date_match and time_match and room_match:
                return {"name": "schedule_meeting", "arguments": json.dumps({
                    "date": date_match.group(1),
                    "time": time_match.group(1),
                    "meeting_room": f"Room {room_match.group(1).capitalize()}"
                })}
        pattern_debug_info["meeting_scheduling"] = re.search(
            r"schedule.?\d{4}-\d{2}-\d{2}.?\d{2}:\d{2}.*?room", query, re.IGNORECASE) is not None

        # Expense balance pattern
        if re.search(r"expense", query):
            emp_match = re.search(r"employee\s*(\d+)", query, re.IGNORECASE)
            if emp_match:
                return {"name": "get_expense_balance", "arguments": json.dumps({
                    "employee_id": int(emp_match.group(1))
                })}
        pattern_debug_info["expense_balance"] = re.search(
            r"expense", query) is not None

        # Performance bonus pattern
        if re.search(r"bonus", query, re.IGNORECASE):
            emp_match = re.search(
                r"emp(?:loyee)?\s*(\d+)", query, re.IGNORECASE)
            year_match = re.search(r"\b(2024|2025)\b", query)
            if emp_match and year_match:
                return {"name": "calculate_performance_bonus", "arguments": json.dumps({
                    "employee_id": int(emp_match.group(1)),
                    "current_year": int(year_match.group(1))
                })}
        pattern_debug_info["performance_bonus"] = re.search(
            r"bonus", query, re.IGNORECASE) is not None

        # Office issue pattern
        if re.search(r"(office issue|report issue)", query, re.IGNORECASE):
            code_match = re.search(
                r"(issue|number|code)\s*(\d+)", query, re.IGNORECASE)
            dept_match = re.search(
                r"(in|for the)\s+(\w+)(\s+department)?", query, re.IGNORECASE)
            if code_match and dept_match:
                return {"name": "report_office_issue", "arguments": json.dumps({
                    "issue_code": int(code_match.group(2)),
                    "department": dept_match.group(2).capitalize()
                })}
        pattern_debug_info["office_issue"] = re.search(
            r"(office issue|report issue)", query, re.IGNORECASE) is not None

        raise HTTPException(
            status_code=400, detail=f"Could not parse query: {q}")

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse query: {q}. Error: {str(e)}. Pattern matches: {pattern_debug_info}"
        )


if __name__ == "__main__":
    import uvicorn

    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

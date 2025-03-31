# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi", "uvicorn"]
# ///
# uv header script
import json
import os
import pickle
from typing import Optional

import aiofiles
import numpy as np
from fastapi import Body, FastAPI, File, Form, Request, UploadFile
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

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("stored_data.pkl", "rb") as f:
    data = pickle.load(f)
    stored_data = data["stored_data"]
    stored_embeddings = data["stored_embeddings"]


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
        return ga_2_9(file)

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
        return ga_3_7()

    elif map_["best_answer"] == "ga_3_8":
        return ga_3_8()
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

    map_ = find_best_match(question, stored_data, stored_embeddings)

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

if __name__ == "__main__":
    import uvicorn

    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

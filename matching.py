import json
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
local_model_path = "./models/all-MiniLM-L6-v2"
model = SentenceTransformer(local_model_path)

# Load stored embeddings


def find_best_match(question, stored_data, stored_embeddings):
    """
    Finds the best matching stored question for a given query using cosine similarity.
    """

    input_embedding = model.encode(question).reshape(1, -1)
    similarities = cosine_similarity(input_embedding, stored_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    return dict({
        "input_question": question,
        "best_match_question": stored_data[best_match_idx]["question"],
        "best_answer": stored_data[best_match_idx]["answer"],
        "similarity_score": similarities[best_match_idx]
    })


def test_():
    test_cases = [
        {
            "question": "What is the output of code -s?",
            "expected_answer": "ga_1_1"
        },
        {
            "question": "Running uv run --with httpie -- https [URL] sends a HTTPS request to the URL.\nSend a HTTPS request to https://httpbin.org/get with the URL encoded parameter email set to 22f3002248@ds.study.iitm.ac.in\nWhat is the JSON output of the command? (Paste only the JSON body, not the headers)",
            "expected_answer": "ga_1_2"
        },
        {
            "question": "run npx -y prettier@3.4.2 README.md | sha256sum.\nWhat is the output of the command?",
            "expected_answer": "ga_1_3"
        },
        {
            "question": "Type this formula into Google Sheets. (It won't work in Excel)\n=SUM(ARRAY_CONSTRAIN(SEQUENCE(100, 100, 15, 7), 1, 10))\nWhat is the result?",
            "expected_answer": "ga_1_4"
        },
        {
            "question": "Type this formula into Excel.\nNote: This will ONLY work in Office 365.\n=SUM(TAKE(SORTBY({5,9,3,2,14,4,3,1,1,13,2,12,12,11,12,6}, {10,9,13,2,11,8,16,14,7,15,5,4,6,1,3,12}), 1, 9))\nWhat is the result?\nNote: If you get #NAME? you have the wrong version of Excel. Find a friend for whom this works.",
            "expected_answer": "ga_1_5"
        },
        {
            "question": "Find the hidden input field's value in the given HTML snippet. What is the extracted value?",
            "expected_answer": "ga_1_6"
        },
        {
            "question": "Count the Wednesdays between {{1990-01-01 to 2020-12-31}}, including both endpoints.",
            "expected_answer": "ga_1_7"
        },
        {
            "question": "Extract extract.csv from {{q-extract-csv-zip.zip}} and find the value in the 'answer' column.",
            "expected_answer": "ga_1_8"
        },
        {
            "question": "Sort the given JSON by age, then by name. Return the sorted JSON as a single line.",
            "expected_answer": "ga_1_9"
        },
        {
            "question": "Convert the contents of {{q-multi-cursor-json.txt}} into a valid JSON object. Compute its hash at tools-in-data-science.pages.dev/jsonhash. What is the hash?",
            "expected_answer": "ga_1_10"
        },
        {
            "question": "Find all <div> elements with class foo and sum their data-value attributes. What is the sum?",
            "expected_answer": "ga_1_11"
        },
        {
            "question": "Extract and process {{q-unicode-data.zip}} to sum values of symbols matching {{† OR Š OR …}}.",
            "expected_answer": "ga_1_12"
        },
        {
            "question": "Create a GitHub repository and push email.json with {{\"email\": \"random_email@example.com\"}}. Provide the raw file URL.",
            "expected_answer": "ga_1_13"
        },
        {
            "question": "Unzip {{q-replace-across-files.zip}}, replace all occurrences of IITM with IIT Madras, and compute the sha256sum using cat * | sha256sum. What is the result?",
            "expected_answer": "ga_1_14"
        },
        {
            "question": "Extract {{q-list-files-attributes.zip}} and list all files with sizes and dates. Find the total size of files >= {{10000}} bytes and modified after {{Wed, 15 Aug, 2012, 5:45 pm IST}}.",
            "expected_answer": "ga_1_15"
        },
        {
            "question": "Extract {{q-move-rename-files.zip}}, move all files into a single folder, rename digits, and compute grep . * | LC_ALL=C sort | sha256sum. What is the result?",
            "expected_answer": "ga_1_16"
        },
        {
            "question": "Extract {{q-compress-files.zip}} and compare a.txt and b.txt. How many lines differ?",
            "expected_answer": "ga_1_17"
        },
        {
            "question": "In a SQLite tickets table, find the total sales for 'Gold' tickets (case-insensitive) as SUM(Units * Price). What is the sum?",
            "expected_answer": "ga_1_18"
        }
    ]
    with open("stored_data.pkl", "rb") as f:
        data = pickle.load(f)
        stored_data = data["stored_data"]
        stored_embeddings = data["stored_embeddings"]
    # Run test cases
    count = 0
    for i, test in enumerate(test_cases, 1):
        result = find_best_match(
            test["question"], stored_data, stored_embeddings)
        print(f"Test Case {i}:")
        print("Input Question:", test["question"])
        print("Expected Answer:", test["expected_answer"])
        print("Predicted Answer:", result["best_answer"])
        print("Similarity Score:", result["similarity_score"])
        if result["best_answer"] == test["expected_answer"]:
            count += 1
        print("Match Status:",
              "✅ PASS" if result["best_answer"] == test["expected_answer"] else "❌ FAIL")
        print("-" * 80)
    print(f"✅ Passed Test Cases: {count} / {len(test_cases)} ✅")
    print("-" * 80)


# test_()

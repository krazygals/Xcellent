import json
import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from rapidfuzz import fuzz, process

# Flask App Setup
app = Flask(__name__)
from flask_cors import CORS
CORS(app, origins=[
    "https://www.xcellentupload.com",
    "https://xcellentupload.com",
    "https://xcellent-frontend-bice.vercel.app",
    "http://localhost:3000"
])

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def home():
    return "Backend is running!"

@app.route("/upload", methods=["POST"])
def upload_file():
    print("🔍 Incoming request data:", request.data)  # Logs the raw request data
    print("🔍 Request headers:", request.headers)
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    
    try:
        # ✅ Handle both JSON and form-data for columns
        user_defined_columns = []
        if request.is_json:
            user_defined_columns = request.json.get("columns", [])
        else:
            user_defined_columns = json.loads(request.form.get("columns", "[]"))  

        if not user_defined_columns:
            return jsonify({"error": "Desired column names are required"}), 400

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Find header row
        header_row = find_header_row(file_path)
        df = pd.read_excel(file_path, engine="openpyxl", header=header_row)

        # Ensure all column names are strings
        df.columns = df.columns.astype(str)

        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.match(r"Unnamed: \d+")]

        print("📥 Uploaded Excel Columns:", df.columns.tolist())
        print("📤 User-defined SQL Columns (raw):", user_defined_columns)

        # Run Fuzzy Matching
        matched_columns = match_columns(df.columns.tolist(), user_defined_columns)

        return jsonify({
            "message": "File uploaded and column matches suggested!",
            "processed_file_path": file_path,
            "suggested_matches": matched_columns
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/match-columns', methods=["POST"])
def match_user_columns():
    try:
        data = request.get_json()
        user_defined_columns = data.get("user_columns", [])

        # Simulate a dummy uploaded Excel file for test-matching
        dummy_uploaded_columns = [
            "SECTIONS_ID", "SECTION_NAME", "FACULTY_NAME",
            "ENROLLED", "START_TIME1", "END_TIME1"
        ]

        matched_columns = match_columns(dummy_uploaded_columns, user_defined_columns)

        return jsonify({
            "message": "Column matches suggested!",
            "suggested_matches": matched_columns
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def find_header_row(file_path):
    """Automatically detects the first row with actual data and uses it as headers"""
    df = pd.read_excel(file_path, engine="openpyxl", header=None, dtype=str)

    for i, row in df.iterrows():
        row_values = row.dropna().astype(str).tolist()
        if len(row_values) > 3 and not any("Applied filters" in val for val in row_values):
            return i

    return 0  # Default to first row if no valid header found

def clean_column_name(name):
    """Normalize column names for better matching."""
    name = name.lower().strip()  
    name = re.sub(r'[^a-z0-9]', ' ', name)  
    name = re.sub(r'\s+', ' ', name)  
    return name.strip()

from sentence_transformers import SentenceTransformer, util

# Load the model once (top of your app.py)
model = SentenceTransformer('all-MiniLM-L6-v2')

from sentence_transformers import util

def clean_col(col):
    return col.lower().replace("_", " ").strip()

def match_columns(uploaded_columns, user_defined_columns, threshold=0.6):
    """Matches uploaded columns to user-defined columns using semantic similarity via Sentence Transformers."""
    matched_columns = {}

    # Clean both sets
    uploaded_clean = [clean_col(col) for col in uploaded_columns]
    user_clean = [clean_col(col) for col in user_defined_columns]

    # Encode
    uploaded_embeddings = model.encode(uploaded_clean, convert_to_tensor=True)
    user_embeddings = model.encode(user_clean, convert_to_tensor=True)

    # Match each user column to the closest uploaded one
    for i, user_col in enumerate(user_clean):
        similarities = util.cos_sim(user_embeddings[i], uploaded_embeddings)[0]
        best_match_idx = similarities.argmax().item()
        score = similarities[best_match_idx].item()

        original_user_col = user_defined_columns[i]
        original_uploaded_col = uploaded_columns[best_match_idx]

        # Log for debugging
        print(f"{original_user_col} → {original_uploaded_col} | Score: {score:.4f}")

        if score > threshold:
            matched_columns[original_user_col] = original_uploaded_col
        else:
            matched_columns[original_user_col] = "❌ No match found"

    return matched_columns

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
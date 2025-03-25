import json
import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from rapidfuzz import fuzz, process

# Flask App Setup
app = Flask(__name__)
CORS(app, origins=["https://www.xcellentapp.com", "http://localhost:3000"])  # Include localhost for testing React locally

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def home():
    return "Backend is running!"

@app.route("/upload", methods=["POST"])
def upload_file():
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

        # Run Fuzzy Matching
        matched_columns = match_columns(df.columns.tolist(), user_defined_columns)

        return jsonify({
            "message": "File uploaded and column matches suggested!",
            "processed_file_path": file_path,
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

    manual_mappings = {
        "crsn": ["sections_id", "section", "crn"],  # ✅ Maps to "SECTIONS_ID"
        "course_no": ["section_name", "course id", "class id"],  # ✅ Maps to "SECTION_NAME"
        "teach_name": ["faculty_name", "instructor", "faculty", "teach name"],  # ✅ Maps to "FACULTY_NAME"
        "enrollment": ["enrolled", "students"],  # ✅ Maps to "ENROLLED"
        "begin_time": ["start time", "begin time", "start_time1"],  # ✅ Maps to "START_TIME1"
        "end_time": ["end time", "finish time", "end_time1"]  # ✅ Maps to "END_TIME1"
    }

def match_columns(uploaded_columns, user_defined_columns):
    """Matches uploaded columns to user-defined columns using manual mappings first, then fuzzy logic."""
    matched_columns = {}

    manual_mappings = {
        "crsn": ["sections_id", "section", "crn"],  # ✅ Maps to "SECTIONS_ID"
        "course_no": ["section_name", "course id", "class id"],  # ✅ Maps to "SECTION_NAME"
        "teach_name": ["faculty_name", "instructor", "faculty", "teach name"],  # ✅ Maps to "FACULTY_NAME"
        "enrollment": ["enrolled", "students"],  # ✅ Maps to "ENROLLED"
        "begin_time": ["start time", "begin time", "start_time1"],  # ✅ Maps to "START_TIME1"
        "end_time": ["end time", "finish time", "end_time1"]  # ✅ Maps to "END_TIME1"
    }
    
    # ✅ Normalize uploaded column names
    uploaded_columns_cleaned = {clean_column_name(col): col for col in uploaded_columns}
    user_defined_cleaned = [clean_column_name(col) for col in user_defined_columns]

    for user_col in user_defined_cleaned:
        found_match = None

        # ✅ Manual Mapping Check (Now Uses Correct Lookup)
        for standard_col, synonyms in manual_mappings.items():
            if clean_column_name(user_col) == clean_column_name(standard_col):  # ✅ Direct match with key
                for uploaded_cleaned, uploaded_original in uploaded_columns_cleaned.items():
                    if uploaded_cleaned in [clean_column_name(x) for x in synonyms]:  # ✅ Check synonyms
                        found_match = uploaded_original
                        break
            if found_match:
                break  # ✅ Exit loop once manual match is found

        # ✅ Fuzzy Match as Fallback (Only If Manual Mapping Fails)
        if not found_match:
            match_result = process.extractOne(user_col, list(uploaded_columns_cleaned.keys()), scorer=fuzz.ratio)
            if match_result:  # Ensure we have a valid match
                best_match, score = match_result[:2]  # Extract only 2 values
                if score > 70:
                    found_match = uploaded_columns_cleaned[best_match]

        # ✅ Assign the best match found
        matched_columns[user_col] = found_match if found_match else "❌ No match found"

    return matched_columns

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

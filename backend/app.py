import json
import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
import faiss # type: ignore
print("FAISS version:", faiss.__version__)
import numpy as np
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import psycopg2 # type: ignore
from flask import send_file
from io import BytesIO


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

faiss_index = None
faiss_column_names = []

@app.route("/upload", methods=["POST"])
@cross_origin(origins=["https://xcellentupload.com", "https://www.xcellentupload.com"])
def upload_file():
    print("🔍 Request method:", request.method)
    print("🔍 Request content-type:", request.content_type)
    print("🔍 request.files:", request.files)
    print("🔍 request.form:", request.form)
    print("🔍 request.json:", request.get_json(silent=True))
    print("🔍 Incoming request data:", request.data)  # Logs the raw request data
    print("🔍 Request headers:", request.headers)
    global faiss_index, faiss_column_names
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    # Guard against missing or empty columns input
    if not request.form.get("columns") and not request.is_json:
        return jsonify({"error": "Missing or invalid column input"}), 400

    
    try:
        # ✅ Handle both JSON and form-data for columns
        user_defined_columns = []
        if request.is_json:
            user_defined_columns = request.json.get("columns", [])
        else:
            raw_columns_input = request.form.get("columns", "")
            user_defined_columns = parse_user_columns(raw_columns_input)

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

        detected_headers = df.columns.tolist()

        print("📥 Uploaded Excel Columns:", detected_headers)
        print("📤 User-defined SQL Columns (raw):", user_defined_columns)

        # 🆕 Build FAISS index
        embeddings = model.encode(detected_headers)
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(embeddings))
        faiss_column_names = detected_headers

        # Keep fuzzy matching (if needed)
        matched_columns = match_columns(detected_headers, user_defined_columns)

        return jsonify({
            "message": "File uploaded and FAISS index built!",
            "processed_file_path": file_path,
            "suggested_matches": matched_columns,
            "detected_headers": detected_headers
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def jaccard_similarity(str1, str2):
    vec = CountVectorizer(binary=True).fit_transform([str1, str2])
    return jaccard_score(vec.toarray()[0], vec.toarray()[1])

@app.route('/api/match-columns', methods=["POST"])
def match_user_columns():
    global faiss_index, faiss_column_names

    data = request.get_json()
    user = data.get("user_columns", [])

    def preprocess(column):
        return column.replace('_', ' ').lower().strip()

    user = [preprocess(col) for col in user]

    matches = []
    used_recognized_indices = set()

    if faiss_index is not None and faiss_column_names:
        user_embeddings = model.encode(user)
        D, I = faiss_index.search(np.array(user_embeddings), k=1)

        for idx, user_col in enumerate(user):
            best_match_idx = I[idx][0]
            if best_match_idx in used_recognized_indices:
                continue

            faiss_col = faiss_column_names[best_match_idx].strip()
            distance = D[idx][0]
            faiss_similarity = 1 / (1 + distance)
            
            levenshtein_score = fuzz.ratio(user_col, preprocess(faiss_col)) / 100.0
            jaccard = jaccard_similarity(user_col, preprocess(faiss_col))

            combined_score = max(faiss_similarity, levenshtein_score)

            print(f"[MATCH] '{user_col}' → '{faiss_col}' | FAISS: {faiss_similarity:.4f}, Levenshtein: {levenshtein_score:.4f}, Combined: {combined_score:.4f}")

            if combined_score > 0.6:  # Adjust as needed
                matches.append({'from': faiss_col, 'to': user_col})

    return jsonify(matches=matches)

    rec_embed = model.encode(recognized, convert_to_tensor=True)
    user_embed = model.encode(user, convert_to_tensor=True)

    matches = []
    if faiss_index is not None and faiss_column_names:
            user_embeddings = model.encode(user)
            D, I = faiss_index.search(np.array(user_embeddings), k=1)
            for idx, user_col in enumerate(user):
                best_match_idx = I[idx][0]
                distance = D[idx][0]
                similarity_score = 1 / (1 + distance)  # Convert L2 distance to a similarity score

                if similarity_score > 0.6:  # Adjust threshold if needed
                    matches.append({
                        "from": faiss_column_names[best_match_idx],
                        "to": user_col
                    })

    return jsonify(matches=matches)

def find_header_row(file_path):
    """Automatically detects the first row with actual data and uses it as headers"""
    df = pd.read_excel(file_path, engine="openpyxl", header=None, dtype=str)

    for i, row in df.iterrows():
        row_values = row.dropna().astype(str).tolist()
        if len(row_values) > 3 and not any("Applied filters" in val for val in row_values):
            return i

    return 0  # Default to first row if no valid header found

def get_db_connection():
    conn = psycopg2.connect(
        dbname="xcellent",
        user="oriana",  # Replace with your actual username
        password="candy123",  # Or remove if not needed
        host="localhost"
    )
    return conn

def fallback_db_match(user_column):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT matched_column FROM validated_matches WHERE user_column = %s",
        (user_column.lower(),)
    )
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result:
        return result[0]  # matched_column
    return None

def parse_user_columns(raw_input):
    # First, check if it's JSON formatted (like a list of columns)
    try:
        columns = json.loads(raw_input)
        if isinstance(columns, list):
            return [col.strip() for col in columns if col.strip()]
    except json.JSONDecodeError:
        pass  # Not JSON, continue...

    # If not JSON, handle as raw text (SQL or Excel)
    lines = raw_input.strip().split('\n')

    # Check if it's likely SQL output (detects if lines have spaces or pipes separating them)
    if any(re.search(r'\s{2,}|\|', line) for line in lines):
        # SQL paste: extract first 'column-like' word from each line
        columns = []
        for line in lines:
            parts = re.split(r'\s{2,}|\|', line)
            if parts and parts[0].strip():
                columns.append(parts[0].strip())
        return columns

    else:
        # Excel paste (tab-separated or comma-separated)
        return [col.strip() for col in re.split(r'[\t,]', raw_input) if col.strip()]

def clean_column_name(name):
    """Normalize column names for better matching."""
    name = name.lower().strip()  
    name = re.sub(r'[^a-z0-9]', ' ', name)  
    name = re.sub(r'\s+', ' ', name)  
    return name.strip()

def clean_col(col):
    return col.lower().replace("_", " ").strip()

def match_columns(uploaded_columns, user_defined_columns, threshold=0.6):
    matched_columns = {}
    matched_uploaded_cols = set()  # Lock recognized columns

    # Clean columns
    uploaded_clean = [clean_col(col) for col in uploaded_columns]
    user_clean = [clean_col(col) for col in user_defined_columns]

    # Encode embeddings
    uploaded_embeddings = model.encode(uploaded_clean, convert_to_tensor=True)
    user_embeddings = model.encode(user_clean, convert_to_tensor=True)

    # For each user column
    for i, user_col in enumerate(user_clean):
        similarities = util.cos_sim(user_embeddings[i], uploaded_embeddings)[0]

        # Mask out used recognized columns
        for idx in matched_uploaded_cols:
            similarities[idx] = -1

        best_match_idx = similarities.argmax().item()
        score = similarities[best_match_idx].item()

        original_user_col = user_defined_columns[i]
        original_uploaded_col = uploaded_columns[best_match_idx]

        # If AI match score is good & not reused
        if score > threshold and best_match_idx not in matched_uploaded_cols:
            matched_columns[original_user_col] = original_uploaded_col
            matched_uploaded_cols.add(best_match_idx)  # Lock recognized column
        else:
            # Fallback to DB if AI fails
            fallback = fallback_db_match(original_user_col)
            if fallback and clean_col(fallback) in uploaded_clean:
                matched_columns[original_user_col] = fallback
                matched_uploaded_cols.add(uploaded_clean.index(clean_col(fallback)))  # Lock fallback
                print(f"[FALLBACK] '{original_user_col}' → '{fallback}'")
            else:
                matched_columns[original_user_col] = "❌ No match found"
                print(f"[NO MATCH] '{original_user_col}' could not be matched.")
        print("[FINAL MATCHED COLUMNS]", matched_columns)
    return matched_columns

@app.route("/api/export", methods=["POST"])
def export_data():
    data = request.get_json()
    matches = data.get("matches", [])
    file_path = data.get("file_path")
    export_format = data.get("format", "xlsx")

    # 🟣 Original match dict (recognized → cleaned)
    match_dict = {m['from']: m['to'] for m in matches}

    # 🟣 Load Excel with original column names
    header_row = find_header_row(file_path)
    df = pd.read_excel(file_path, engine="openpyxl", header=header_row)
    print("[DEBUG] file_path received:", file_path)
    print("[DEBUG] DataFrame original columns:", df.columns.tolist())

    # 🟣 Extract only recognized columns (no cleaning here!)
    selected_cols = list(match_dict.keys())
    print("[DEBUG] Selected columns (from matches):", selected_cols)

    try:
        output_df = df[selected_cols].copy()
    except KeyError as e:
        return jsonify({"error": f"Column mismatch: {e}"}), 400

    # 🟣 Rename to cleaned column names
    output_df.rename(columns=match_dict, inplace=True)
    print("[DEBUG] Output columns after renaming:", output_df.columns.tolist())

    # 🟣 Preview logic
    preview_data = {
        "headers": list(output_df.columns),
        "rows": output_df.head(5).values.tolist()
    }

    # 🟢 Handle export formats
    if export_format == "preview":
        return jsonify(preview=preview_data)
    else:
        output = BytesIO()
        if export_format == "csv":
            output_df.to_csv(output, index=False)
            output.seek(0)
            return send_file(output, mimetype='text/csv', download_name="exported_data.csv", as_attachment=True)
        elif export_format == "xlsx":
            output_df.to_excel(output, index=False, engine="openpyxl")
            output.seek(0)
            return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', download_name="exported_data.xlsx", as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
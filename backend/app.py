from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from rapidfuzz import process

app = Flask(__name__)
CORS(app)  # Allow frontend requests

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

MERGED_DATA_FILE = os.path.join(UPLOAD_FOLDER, "merged_data.xlsx")

@app.route("/")
def home():
    return "Backend is running!"

# ✅ Dictionary of standardized column names with possible variations
COLUMN_MAPPINGS = {
    "higher_ed": {
        "crsn": ["CRN", "Course ID", "Section Number"],
        "dept": ["Department", "Dept"],
        "course_no": ["Course Number", "Course No", "Class Number"],
        "sect": ["Section", "Sect"],
        "subject": ["Subject", "Course Subject"],
        "instructor": ["Instructor", "Faculty Name", "Professor"],
        "enrollment": ["Enrollment", "Students Enrolled", "Total Enrollment"],
        "begin_time": ["Start Time", "Begin Time"],
        "end_time": ["End Time", "Finish Time"],
        "bld_no": ["Building Number", "Bld No"],
        "rm_no": ["Room Number", "Rm No"]
    }
}

# ✅ Higher Ed Required Columns
HIGHER_ED_COLUMNS = [
    "crsn", "dept", "course_no", "sect", "subject", "mon", "tue", "wed", "thur",
    "fri", "sat", "sun", "week_day", "week_day_bu", "begin_time", "bgn_hrs",
    "bgn_min", "bgn_dec_time", "end_time", "end_hrs", "end_min", "end_dec_time",
    "time_in_class", "bld_no", "bld", "bld_id", "rm_no", "capacity_sch",
    "capacity_astra", "occ_load", "occ_load_final", "delivery_method", "campus",
    "campus_name", "teach_name", "begin_date", "end_date", "weeks_in_class",
    "days_in_class", "enrollment_capacity", "enrollment", "dch", "wsch_old",
    "wsch", "cr_hrs", "crse_title", "room_type", "room_type_bu", "semester",
    "FCIMCode", "notes", "idinc", "credit hour production"
]

def find_header_row(file_path):
    """ Dynamically finds the first row that contains actual column names """
    df = pd.read_excel(file_path, engine="openpyxl", header=None, dtype=str)

    for i, row in df.iterrows():
        row_values = row.dropna().astype(str).tolist()  # Remove NaN and convert to strings

        # ✅ If the row contains at least 2 non-empty values, assume it's the header
        if len(row_values) >= 2:
            return i  # Use this row as the header

    return 0  # Default to first row if no match is found


# ✅ AI-Based Column Matching
from rapidfuzz import process

from rapidfuzz import process

def match_columns(file_type, uploaded_columns):
    """ Matches uploaded columns to standardized column names using RapidFuzz and debugging output """
    matched_columns = {}

    if file_type not in COLUMN_MAPPINGS:
        print("❌ ERROR: No column mappings found for file type:", file_type)
        return matched_columns  # No mappings available

    standard_columns = COLUMN_MAPPINGS[file_type]

    print("\n🔍 Uploaded Column Names:", uploaded_columns)  # Debugging Output
    print("🎯 Standard Column Names:", list(standard_columns.keys()))  # Debugging Output

    for uploaded_col in uploaded_columns:
        match_result = process.extractOne(uploaded_col.strip().lower(), [col.lower() for col in sum(standard_columns.values(), [])])

        if match_result:
            best_match, score, *_ = match_result  # Handle extra values
            print(f"📌 Matching: {uploaded_col} → {best_match} (Score: {score})")  # Debugging Output
            
            if score > 70:  # Lowered threshold from 80 to 70 for better flexibility
                matched_col = [key for key, values in standard_columns.items() if best_match.lower() in [v.lower() for v in values]][0]
                matched_columns[uploaded_col] = matched_col
            else:
                print(f"⚠️ No good match found for: {uploaded_col} (Best score: {score})")  # Show failed matches
                matched_columns[uploaded_col] = None
        else:
            print(f"❌ No match found for: {uploaded_col}")  # Show completely failed matches
            matched_columns[uploaded_col] = None

    print("✅ Final Mapped Columns:", matched_columns)  # Debugging Output
    return matched_columns

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    file_type = request.form.get("file_type")  # "higher_ed" or "lower_ed"

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file_type not in ["higher_ed", "lower_ed"]:
        return jsonify({"error": "Invalid file type"}), 400

    if file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        try:
            # Detect the correct header row
            header_row = find_header_row(file_path)

            # Read the Excel file using the detected header row
            df = pd.read_excel(file_path, engine="openpyxl", header=header_row)

            # Remove unnecessary rows above the detected header
            df = df.iloc[header_row:].reset_index(drop=True)

            # Remove unnamed columns
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

            # Strip spaces from column names to clean them
            df.columns = df.columns.str.strip()

            # AI-based column name recognition
            mapped_columns = match_columns(file_type, df.columns)
            df.rename(columns=mapped_columns, inplace=True)

            # Ensure all required columns exist
            for col in HIGHER_ED_COLUMNS:
                if col not in df.columns:
                    df[col] = None  # Add missing columns

            # Reorder columns
            df = df[HIGHER_ED_COLUMNS]

            # Save processed data
            processed_file_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_" + file.filename)
            df.to_excel(processed_file_path, index=False)

            return jsonify({
                "message": "File processed and formatted successfully",
                "processed_file_path": processed_file_path,
                "mapped_columns": mapped_columns
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
def merge_data(new_file_path):
    """ Merge uploaded file with existing dataset """
    new_df = pd.read_excel(new_file_path, engine="openpyxl")

    # If no merged data exists, create a new file
    if not os.path.exists(MERGED_DATA_FILE):
        new_df.to_excel(MERGED_DATA_FILE, index=False)
        return "New dataset created"

    # Load existing merged data
    existing_df = pd.read_excel(MERGED_DATA_FILE, engine="openpyxl")

    # ✅ Merge on room number (rm_no) and course number (crsn)
    merged_df = pd.merge(existing_df, new_df, on=["rm_no", "crsn"], how="outer")

    # Save merged dataset
    merged_df.to_excel(MERGED_DATA_FILE, index=False)
    return "Data merged successfully"

@app.route("/merge", methods=["POST"])
def merge_uploaded_file():
    file_path = request.json.get("file_path")

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "Invalid file path"}), 400

    message = merge_data(file_path)

    return jsonify({"message": message, "merged_data_path": MERGED_DATA_FILE}), 200

if __name__ == "__main__":
    app.run(debug=True)
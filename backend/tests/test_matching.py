from rapidfuzz import process

def match_columns(uploaded_columns, user_defined_columns):
    """Matches uploaded columns to user-defined columns using fuzzy logic with manual overrides"""
    matched_columns = {}

    # ✅ Strong manual mappings to avoid fuzzy mismatches
    manual_mappings = {
        "SECTIONS_ID": "crsn",
        "TERM": "semester",
        "SECTION_NAME": "course_no",
        "TITLE": "crse_title",
        "FACULTY_NAME": "teach_name",
        "START_TIME1": "begin_time",
        "END_TIME1": "end_time",
        "ROOM1": "rm_no",
        "ROOM2": "rm_no",
        "CAPACITY": "enrollment_capacity",
        "ENROLLED": "enrollment",
        "CREDITS": "cr_hrs",
        "START_DATE": "begin_date",
        "END_DATE": "end_date",
        "INSTR_TYPE": "delivery_method",
        "LOCATION": "bld"
    }

    print("🔍 Uploaded Columns:", uploaded_columns)
    print("🎯 Expected User-Defined Columns:", user_defined_columns)

    if not uploaded_columns or not user_defined_columns:
        print("⚠️ No columns available for matching.")
        return matched_columns  # No matching possible

    for user_col in user_defined_columns:
        # ✅ First, check manual mappings
        if user_col in manual_mappings.values():
            matched_columns[user_col] = next(
                (key for key, value in manual_mappings.items() if value == user_col), 
                "❌ No match found"
            )
            print(f"✅ Manual Mapping: {user_col} → {matched_columns[user_col]}")
            continue  # Skip fuzzy matching for these

        # ✅ Fuzzy match the remaining columns
        match_results = process.extract(user_col, uploaded_columns, limit=3)  # Get top 3 matches

        if match_results:
            best_match, best_score = match_results[0]
            print(f"📌 Matching: {best_match} → {user_col} (Score: {best_score})")

            if best_score < 50:
                matched_columns[user_col] = "❌ No match found"
            elif best_score > 85:
                matched_columns[user_col] = best_match
            else:
                second_best = match_results[1] if len(match_results) > 1 else None
                if second_best and second_best[1] > 70:
                    matched_columns[user_col] = f"⚠️ Weak match: {second_best[0]} (Check this)"
                else:
                    matched_columns[user_col] = f"⚠️ Weak match: {best_match} (Check this)"
        else:
            matched_columns[user_col] = "❌ No match found"

    print("✅ Final Suggested Matches:", matched_columns)  # ✅ Debugging Output
    return matched_columns


# ✅ Test Data: Small dataset first
uploaded_columns = [
    "SECTIONS_ID", "TERM", "SECTION_NAME", "TITLE", "FACULTY_NAME", 
    "START_TIME1", "END_TIME1", "ROOM1", "CAPACITY", "ENROLLED", "CREDITS"
]

user_defined_columns = ["crsn", "semester", "course_no", "crse_title", "teach_name", 
                        "begin_time", "end_time", "rm_no", "enrollment_capacity", 
                        "enrollment", "cr_hrs"]

# Run Matching
matched_results = match_columns(uploaded_columns, user_defined_columns)

# ✅ Print Results
print("\n🔎 FINAL MATCHES:")
for key, value in matched_results.items():
    print(f"{key} → {value}")

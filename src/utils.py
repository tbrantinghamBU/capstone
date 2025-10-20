import csv
import os
from datetime import datetime

def log_results_to_csv(results_list, filename="mend_results.csv"):
    """
    Appends the latest result dictionary from results_list to a CSV file.
    - Adds a timestamp column
    - Handles varying keys across runs
    - Avoids duplicate writes if the same result already exists
    - Creates parent directories if they don't exist
    - Ensures Timestamp is the first column
    """
    if not results_list:
        print("No results to log.")
        return
    
    latest_result = results_list[-1].copy()
    latest_result["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    file_exists = os.path.isfile(filename)
    existing_rows, existing_fields = [], []
    
    if file_exists:
        with open(filename, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames
            existing_rows = list(reader)
    
    # Union of old and new fields
    all_fields = list(dict.fromkeys((existing_fields or []) + list(latest_result.keys())))
    
    # Force Timestamp to be first
    if "Timestamp" in all_fields:
        all_fields = ["Timestamp"] + [f for f in all_fields if f != "Timestamp"]
    
    def row_equal(r1, r2):
        return {k: v for k, v in r1.items() if k != "Timestamp"} == \
               {k: v for k, v in r2.items() if k != "Timestamp"}
    
    if any(row_equal(row, latest_result) for row in existing_rows):
        print("Result already logged. Skipping duplicate.")
        return
    
    if file_exists and set(all_fields) != set(existing_fields):
        with open(filename, "w", newline="", encoding="utf-8") as fw:
            writer = csv.DictWriter(fw, fieldnames=all_fields)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)
            writer.writerow(latest_result)
    else:
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(latest_result)
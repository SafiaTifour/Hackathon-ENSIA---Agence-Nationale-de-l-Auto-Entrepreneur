import pandas as pd
import re
from langdetect import detect
import sys

sys.stdout.reconfigure(encoding='utf-8')  # Fix Unicode issues on Windows

# Function to check valid text
def is_valid_text(text):
    text = str(text).strip()
    clean_text = re.sub(r'[^\w\s]', '', text)
    if sum(c.isalpha() for c in clean_text) / max(len(clean_text), 1) < 0.5:
        return False
    try:
        lang = detect(text)
        return lang in ["en", "fr", "ar"]
    except:
        return False

# Load raw data
file_path = "../../data/raw/preprocessed_propositions_copy.xlsx"  # Replace with actual input file
df = pd.read_excel(file_path)

# Debugging: Print column names
print("Columns in DataFrame:", df.columns.tolist())

# Ensure column names are lowercase and stripped
df.columns = df.columns.str.strip().str.lower()

# Validate column existence
if "name" not in df.columns or "description" not in df.columns:
    raise ValueError("❌ Columns 'name' or 'description' not found in the input file.")

# Filter valid suggestions
df_valid = df[df["name"].apply(is_valid_text) & df["description"].apply(is_valid_text)]

# Save valid suggestions
valid_output = "../../data/post_valid/valid_suggested_activities.xlsx"
df_valid.to_excel(valid_output, index=False)

print(f"✔️ Valid suggestions saved in '{valid_output}'.")

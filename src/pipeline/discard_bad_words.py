from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
from langdetect import detect
import numpy as np
import os
import sys

# ✅ Fix Unicode issues (Windows)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ✅ Load Sentence Transformer model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ✅ Load the data
file_path = "../../data/post_valid/valid_suggested_activities.xlsx"

# ✅ Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Error: The file '{file_path}' does not exist.")

try:
    df = pd.read_excel(file_path, engine="openpyxl")  # Ensure engine is correct
except Exception as e:
    raise ValueError(f"🚨 Error reading the input file '{file_path}': {e}")

# ✅ Ensure column names are lowercase and stripped
df.columns = df.columns.str.strip().str.lower()

# ✅ Handle missing values
df.fillna("", inplace=True)

# ✅ Combine name and description columns
if "name" not in df.columns or "description" not in df.columns:
    raise ValueError("❌ Columns 'name' or 'description' not found in the input file.")

selected_activities = np.array(df["name"] + " " + df["description"])  # description

# ✅ Load forbidden words dataset
bad = []
for bad_file in ["../../artifacts/ar.xlsx", "../../artifacts/en.xlsx", "../../artifacts/fr.xlsx"]:
    if not os.path.exists(bad_file):
        print(f"⚠️ Warning: File {bad_file} not found. Skipping...")
        continue
    
    try:
        df_bad = pd.read_excel(bad_file, engine="openpyxl")  # Ensure correct engine
        bad.extend(df_bad.iloc[:, 0].dropna().astype(str).tolist())  # Ensure it's a list of strings
    except FileNotFoundError:
        print(f"⚠️ Warning: File {bad_file} not found. Skipping...")
    except ValueError:
        print(f"⚠️ Warning: {bad_file} is empty or corrupted. Skipping...")
    except Exception as e:
        print(f"🚨 Error reading {bad_file}: {e}")

# ✅ Convert to NumPy array
bad = np.array(bad)

# ✅ Check for forbidden words
if bad.size == 0:
    print("⚠️ No forbidden words found, skipping similarity check.")
else:
    current_embeddings = model.encode(bad, convert_to_tensor=True, normalize_embeddings=True)

    def check_for_bad_word(selected_activities, threshold=0.60):
        filtered_activities = []
        discarded_activities = []

        for activity in selected_activities:
            if not isinstance(activity, str) or activity.strip() == "":
                continue  # Skip empty or non-string values

            new_embedding = model.encode(activity, convert_to_tensor=True, normalize_embeddings=True)
            similarity_scores = util.pytorch_cos_sim(new_embedding, current_embeddings)

            if similarity_scores.numel() == 0:  # Check if similarity_scores is empty
                continue

            max_similarity, nearest_index = similarity_scores.max(dim=1)
            max_similarity = max_similarity.item()
            nearest_activity = bad[nearest_index.item()]

            if max_similarity >= threshold:
                discarded_activities.append(activity)
            else:
                filtered_activities.append(activity)

        return filtered_activities, discarded_activities

    # ✅ Run the filtering function
    filtered_suggestions, discarded_suggestions = check_for_bad_word(selected_activities)

    # ✅ Save results
    try:
        pd.DataFrame(filtered_suggestions, columns=["filtered_activities"]).to_excel(
            "../../data/post_valid/valid_suggested_activities.xlsx", index=False
        )
        pd.DataFrame(discarded_suggestions, columns=["discarded_activities"]).to_excel(
            "../../data/post_valid/discarded_suggestions.xlsx", index=False
        )

        print("✅ Test completed! Unique suggestions saved in 'valid_suggested_activities.xlsx'.")
        print("❌ Discarded suggestions saved in 'discarded_suggestions.xlsx'.")

    except Exception as e:
        print(f"🚨 Error occurred while saving results: {str(e)}")

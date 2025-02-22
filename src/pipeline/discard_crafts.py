from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load filtered suggestions
file_path = "../../data/post_commercial_filter/filtered_suggested_activities.xlsx"
df_filtered = pd.read_excel(file_path)

# Load commercial activities from CSV
commercial_activities_path = "../../artifacts/crafts.csv" 
df_commercial = pd.read_csv(commercial_activities_path, header=None, names=["name"])

# Encode commercial activities
commercial_list = df_commercial["name"].dropna().tolist()  # Remove any NaN values
commercial_embeddings = model.encode(commercial_list, convert_to_tensor=True, normalize_embeddings=True)

# Function to check duplicates and calculate sum_similarity
def check_duplicates(suggestions, threshold=0.60):
    filtered_activities = []
    discarded_activities = []

    for _, row in suggestions.iterrows():
        if pd.isna(row["name"]) or pd.isna(row["description"]):  # Skip rows with NaN values
            continue

        new_embedding = model.encode([row["name"] + " - " + row["description"]], convert_to_tensor=True, normalize_embeddings=True)
        similarity_scores = util.pytorch_cos_sim(new_embedding, commercial_embeddings)

        if similarity_scores.numel() == 0:  # Check if similarity_scores is empty
            continue

        max_similarity, nearest_index = similarity_scores.max(dim=1)
        sum_similarity = similarity_scores.sum().item()  # Sum of all similarity scores

        row_data = row.to_dict()

        if max_similarity.item() >= threshold:
            discarded_activities.append(row_data)
        else:
            filtered_activities.append(row_data)

    return pd.DataFrame(filtered_activities), pd.DataFrame(discarded_activities)

# Run duplicate detection against commercial activities
df_final, df_discarded_external = check_duplicates(df_filtered)

# Sort final data by sum_similarity in descending order
df_final = df_final.sort_values(by="sum_similarity", ascending=True)

# Save outputs
final_output = "../../data/final/final_suggested_activities.xlsx"
discarded_external_output = "../../data/post_crafts_filter/discarded_suggestions.xlsx"

df_final.to_excel(final_output, index=False)
df_discarded_external.to_excel(discarded_external_output, index=False)

print(f"✅ Final suggestions saved in '{final_output}'.")
print(f"❌ Discarded external duplicates saved in '{discarded_external_output}'.")

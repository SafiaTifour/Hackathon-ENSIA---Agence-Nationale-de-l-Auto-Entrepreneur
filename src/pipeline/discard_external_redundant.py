from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load filtered suggestions
file_path = "../../data/post_internal/filtered_suggested_activities.xlsx"
df_filtered = pd.read_excel(file_path)

# Load current activities (external list)
current_activities_path = "../../data/raw/dataset_merge.xlsx"  # Replace with actual file
df_current = pd.read_excel(current_activities_path)

# Encode current activities
current_list = df_current["name"].tolist()
current_embeddings = model.encode(current_list, convert_to_tensor=True, normalize_embeddings=True)

# Function to check duplicates and compute similarity sum
def check_duplicates(suggestions, threshold=0.60):
    filtered_activities = []
    discarded_activities = []

    for _, row in suggestions.iterrows():
        new_embedding = model.encode([row["name"] + " - " + row["description"]], convert_to_tensor=True, normalize_embeddings=True)
        similarity_scores = util.pytorch_cos_sim(new_embedding, current_embeddings)
        sum_similarity = similarity_scores.sum().item()  # Sum of all similarities
        max_similarity, nearest_index = similarity_scores.max(dim=1)
        max_similarity = max_similarity.item()
        nearest_activity = current_list[nearest_index.item()]

        row_data = row.to_dict()
        row_data["sum_similarity"] = sum_similarity


        if max_similarity >= threshold:
            discarded_activities.append(row_data)
        else:
            filtered_activities.append(row_data)

    return pd.DataFrame(filtered_activities), pd.DataFrame(discarded_activities)

# Run external duplicate detection
df_final, df_discarded_external = check_duplicates(df_filtered)

# Save outputs
final_output = "../../data/post_external/filtered_suggested_activities.xlsx"
discarded_external_output = "../../data/post_external/discarded_external_suggestions.xlsx"

df_final.to_excel(final_output, index=False)
df_discarded_external.to_excel(discarded_external_output, index=False)

print(f"✅ Final suggestions saved in '{final_output}'.")
print(f"❌ Discarded external duplicates saved in '{discarded_external_output}'.")

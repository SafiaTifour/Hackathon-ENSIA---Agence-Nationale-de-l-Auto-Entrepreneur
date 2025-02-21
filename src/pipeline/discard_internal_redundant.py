from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load valid suggestions
file_path = "../../data/post_valid/valid_suggested_activities.xlsx"
df_valid = pd.read_excel(file_path)

# Function to remove internal duplicates
def remove_internal_duplicates(suggestions, threshold=0.60):
    unique_suggestions = []
    discarded_internal = []

    suggestion_texts = suggestions["name"] + " - " + suggestions["description"]
    embeddings = model.encode(suggestion_texts.tolist(), convert_to_tensor=True, normalize_embeddings=True)

    for i, text in enumerate(suggestion_texts):
        is_duplicate = False
        new_embedding = embeddings[i].unsqueeze(0)
        similarity_scores = util.pytorch_cos_sim(new_embedding, embeddings)

        for j, score in enumerate(similarity_scores[0]):
            if i != j and score.item() >= threshold:
                is_duplicate = True
                discarded_internal.append(suggestions.iloc[i])
                break

        if not is_duplicate:
            unique_suggestions.append(suggestions.iloc[i])

    return pd.DataFrame(unique_suggestions), pd.DataFrame(discarded_internal)

# Run internal duplicate removal
df_filtered, df_discarded_internal = remove_internal_duplicates(df_valid)

# Save outputs
filtered_output = "../../data/post_internal/filtered_suggested_activities.xlsx"
discarded_internal_output = "../../data/post_internal/discarded_internal_suggestions.xlsx"

df_filtered.to_excel(filtered_output, index=False)
df_discarded_internal.to_excel(discarded_internal_output, index=False)

print(f"✅ Filtered activities saved in '{filtered_output}'.")
print(f"❌ Discarded internal duplicates saved in '{discarded_internal_output}'.")

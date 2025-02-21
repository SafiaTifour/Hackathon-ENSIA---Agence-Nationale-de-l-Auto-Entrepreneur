from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
from langdetect import detect

# Load model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# 1️⃣ load the data 

file_path = "../data/raw/dataset_merge.xlsx"  
df = pd.read_excel(file_path)

# Display the first few rows
print(df.head())

current_activities 

suggested_activities 




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

# Apply filtering
valid_suggestions = [act for act in suggested_activities if is_valid_text(act['name']) and is_valid_text(act['description'])]

# ✅ Internal duplicate removal within suggested activities
def remove_internal_duplicates(suggestions, threshold=0.60):
    unique_suggestions = []
    discarded_internal = []
    
    suggestion_texts = [act['name'] + ' - ' + act['description'] for act in suggestions]
    embeddings = model.encode(suggestion_texts, convert_to_tensor=True, normalize_embeddings=True)
    
    for i, text in enumerate(suggestion_texts):
        is_duplicate = False
        new_embedding = embeddings[i].unsqueeze(0)
        similarity_scores = util.pytorch_cos_sim(new_embedding, embeddings)
        
        for j, score in enumerate(similarity_scores[0]):
            if i != j and score.item() >= threshold:
                is_duplicate = True
                discarded_internal.append(suggestions[i])
                break
        
        if not is_duplicate:
            unique_suggestions.append(suggestions[i])
    
    return unique_suggestions, discarded_internal

# Run internal duplicate removal
valid_suggestions, discarded_internal_suggestions = remove_internal_duplicates(valid_suggestions)
print(f"❌ {len(discarded_internal_suggestions)} internal duplicates removed.")

# ✅ External duplicate checking against current activities
current_list = [act['name'] for act in current_activities]
current_embeddings = model.encode(current_list, convert_to_tensor=True, normalize_embeddings=True)

def check_duplicates(suggestions, threshold=0.60):
    filtered_activities = []
    discarded_activities = []
    
    for activity in suggestions:
        new_embedding = model.encode([activity['name'] + ' - ' + activity['description']], convert_to_tensor=True, normalize_embeddings=True)
        similarity_scores = util.pytorch_cos_sim(new_embedding, current_embeddings)
        max_similarity, nearest_index = similarity_scores.max(dim=1)
        max_similarity = max_similarity.item()
        nearest_activity = current_list[nearest_index.item()]

        if max_similarity >= threshold:
            discarded_activities.append(activity)
        else:
            filtered_activities.append(activity)
    
    return filtered_activities, discarded_activities

# Run duplicate detection with external comparison
filtered_suggestions, discarded_suggestions = check_duplicates(valid_suggestions)
try:
    # Save results to Excel
    pd.DataFrame(filtered_suggestions).to_excel("filtered_suggested_activities_test.xlsx", index=False)
    pd.DataFrame(discarded_suggestions + discarded_internal_suggestions).to_excel("discarded_suggested_activities_test.xlsx", index=False)
    print("************* Discarded")
    print(discarded_suggestions)
    print("************* Discarded2")
    print(discarded_internal_suggestions)
    print("✅ Test completed! Unique suggestions saved in 'filtered_suggested_activities_test.xlsx'.")
    print("❌ Discarded suggestions saved in 'discarded_suggested_activities_test.xlsx'.")

except Exception as e:
        print(f"�� Error occurred while saving results: {str(e)}")
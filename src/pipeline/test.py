from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
from langdetect import detect

# Load model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# 1️⃣ load the data 

# src\pipeline\test.py
# data\raw\dataset_merge.xlsx

file_path = "data/raw/dataset_merge.xlsx"  
df = pd.read_excel(file_path)

# Display the first few rows
print(df.head())

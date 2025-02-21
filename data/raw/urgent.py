import pandas as pd

# Read the Excel file
file_path = r"C:\Users\user\Desktop\Hakathon\data\raw\preprocessed_propositions.xlsx"  # Change this to your actual file path
df = pd.read_excel(file_path)

# Keep only the first 200 rows
df_limited = df.head(200)

# Save to a new Excel file
output_file = "preprocessed_propositions_copy.xlsx"
df_limited.to_excel(output_file, index=False)


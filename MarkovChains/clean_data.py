# clean_data.py
"""Markov Chains.
Sophia Carter
4/27/24
"""

import pandas as pd

#reading in data
file_path = "Advertisement_Transcripts_deduped_edited.xlsx"
df =  pd.read_excel(file_path)


# Extract the 4th column (index 3 because indices are zero-based)
ad_scripts = df.iloc[:, 3]  # Gets the entire 4th column

# Remove newline characters from each row
ad_scripts_cleaned = ad_scripts.str.replace(r'\n', ' ', regex=True)  # Replace newline with a space

# Create a .txt file with the cleaned data
txt_file_path = 'ad_scripts.txt'  # Path for the .txt file
ad_scripts_cleaned.to_csv(txt_file_path, sep='\n', index=False, header=False)  # Save to text file
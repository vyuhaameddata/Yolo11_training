import pandas as pd

# Load the Excel file (modify path as needed)
excel_path = r'E:\archive_Shreyan\FPfeb25\FP.xlsx'
df = pd.read_excel(excel_path, dtype=str)  # Read as strings to avoid NaN issues

# Extract column names excluding the first column
column_names_list = df.columns[1:].tolist()
column_names_dict = {name: i for i, name in enumerate(column_names_list)}

# Print results
print("Column Names as List:", column_names_list)
print("Column Names as Dictionary:", column_names_dict)
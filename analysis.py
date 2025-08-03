# Import the pandas library, which is essential for working with dataframes
import pandas as pd

print("Attempting to load the dataset...")

# Use a try-except block for robust error handling
try:
    # Load the dataset from the CSV file into a pandas DataFrame
    df = pd.read_csv("payment_fraud.csv")
    
    # Print a success message
    print("Dataset loaded successfully!")
    
    # --- Initial Data Inspection ---
    
    # Print the first 5 rows of the dataframe to get a first look at the data
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Print a concise summary of the dataframe, including data types and non-null values
    print("\nDataset Info:")
    df.info(verbose=False) # verbose=False makes the output cleaner for large numbers of columns

except FileNotFoundError:
    print("\n--- ERROR ---")
    print("The file 'payment_fraud.csv' was not found in the current directory.")
    print("Please make sure you have downloaded the dataset and placed it in the 'fraud-detection-project' folder.")
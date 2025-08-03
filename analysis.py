# =============================================================================
# Step 0: Import Libraries
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Step 1: Load and Inspect the Data
# =============================================================================
print("Attempting to load the dataset...")

# Use a try-except block for robust error handling
try:
    # Load the dataset from the CSV file into a pandas DataFrame
    df = pd.read_csv("payment_fraud.csv")
    
    # Print a success message
    print("Dataset loaded successfully!")
    
    # --- Initial Data Inspection ---
    
    # Print the first 5 rows to get a first look at the data
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Print a concise summary of the dataframe
    print("\nDataset Info:")
    df.info(verbose=False)

except FileNotFoundError:
    print("\n--- ERROR ---")
    print("The file 'payment_fraud.csv' was not found in the current directory.")
    print("Please make sure you have downloaded the dataset and placed it in the 'fraud-detection-project' folder.")
    # Exit the script if the file is not found
    exit()

# =============================================================================
# Step 2: Exploratory Data Analysis (EDA)
# =============================================================================
# This entire section will only run if the DataFrame 'df' was successfully loaded
if 'df' in locals():

    print("\n--- Starting Exploratory Data Analysis ---")
    
    # --- Question 1: How balanced is the dataset? ---
    print("\nDistribution of Fraudulent vs. Non-Fraudulent Transactions:")
    fraud_counts = df['isFraud'].value_counts()
    print(fraud_counts)

    # Visualize the class distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(
        x='isFraud',
        data=df,
        hue='isFraud', # Assigning hue prevents a warning
        palette=['#4169E1', '#FF6347'], # Using valid hex codes: royalblue and tomato
        legend=False # We don't need a legend for this plot
    )
    plt.title('Transaction Class Distribution', fontsize=16)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Transactions', fontsize=12)
    plt.xticks([0, 1], ['Non-Fraudulent', 'Fraudulent'])
    # In a real analysis, you might save the plot to a file instead of showing it
    # plt.savefig('fraud_distribution.png')
    
    # --- Question 2: Are certain transaction types more prone to fraud? ---
    print("\nFraud Counts by Transaction Type:")
    # Using crosstab for a clear, side-by-side comparison
    type_fraud_dist = pd.crosstab(df['type'], df['isFraud'])
    print(type_fraud_dist)

    # --- Key Insight ---
    print("\n!!! Key Insight: Fraud only occurs in 'TRANSFER' and 'CASH_OUT' transactions. !!!")
    
else:
    print("\nDataframe 'df' not found. Please ensure the data loading section ran correctly.")
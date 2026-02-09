import pandas as pd

# Load your CSV
df = pd.read_csv('q6_results.csv')

print("="*80)
print("COLUMN NAMES IN YOUR CSV FILE:")
print("="*80)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n" + "="*80)
print("FIRST 3 ROWS OF DATA:")
print("="*80)
print(df.head(3))

print("\n" + "="*80)
print("DATA TYPES:")
print("="*80)
print(df.dtypes)

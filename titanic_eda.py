import pandas as pd
import numpy as np
import seaborn as sns

# Step 1: Three questions to guide the analysis
questions = [
    "1. Did women survive at a higher rate than men?",
    "2. Did passenger class (pclass) affect survival rates?",
    "3. How did survival rates differ across age bands?"
]
print("=== RESEARCH QUESTIONS ===")
for q in questions:
    print(q)

# Step 2: Load and inspect the dataset
df = sns.load_dataset("titanic")

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- FIRST 5 ROWS (head) ---")
print(df.head())

print("\n--- DATASET INFO (info) ---")
print(df.info())

print("\n--- STATISTICAL SUMMARY (describe) ---")
print(df.describe(include="all"))

# Step 3: Find missing values
print("\n--- MISSING VALUES BEFORE CLEANING ---")
print(df.isna().sum())

# Step 4: Fix missing values with clear reasons
# - 'age': grouped median by pclass and sex accounts for demographic age differences without skew from outliers
# - 'embarked': mode fills the few missing port entries with the most common embarkation port
# - 'deck': dropped because the vast majority of entries are missing
df_clean = df.copy()
df_clean["age"] = df_clean["age"].fillna(
    df_clean.groupby(["pclass", "sex"])["age"].transform("median")
)
df_clean["embarked"] = df_clean["embarked"].fillna(df_clean["embarked"].mode()[0])
df_clean = df_clean.drop(columns=["deck"])

print("\n--- MISSING VALUES AFTER CLEANING ---")
print(df_clean.isna().sum())

# Step 5: Check duplicates and data types
print(f"\nDuplicate rows count: {df_clean.duplicated().sum()}")
df_clean = df_clean.drop_duplicates()
print(f"Shape after removing duplicates: {df_clean.shape}")

# Step 6: Group and compare survival rates (sex, pclass, and age_band)
df_clean["age_band"] = pd.cut(
    df_clean["age"],
    bins=[0, 12, 18, 40, 60, 100],
    labels=["Child", "Teen", "Adult", "Middle-aged", "Senior"]
)

survival_by_sex = (df_clean.groupby("sex")["survived"].mean() * 100).round(1).sort_values(ascending=False)
survival_by_class = (df_clean.groupby("pclass")["survived"].mean() * 100).round(1).sort_values(ascending=False)
survival_by_age = (
    df_clean.groupby("age_band", observed=True)["survived"].mean() * 100
).round(1).sort_values(ascending=False)

print("\n--- SURVIVAL RATE (%) BY SEX ---")
print(survival_by_sex)

print("\n--- SURVIVAL RATE (%) BY CLASS ---")
print(survival_by_class)

print("\n--- SURVIVAL RATE (%) BY AGE BAND ---")
print(survival_by_age)

# Step 7: One-sentence answers with numbers
print("\n=== SUMMARY ANSWERS ===")
print(
    f"1. Female passengers had a survival rate of {survival_by_sex.get('female', 0)}% "
    f"compared to {survival_by_sex.get('male', 0)}% for male passengers."
)
print(
    f"2. Survival rates varied by class, with 1st class at {survival_by_class.get(1, 0)}%, "
    f"2nd class at {survival_by_class.get(2, 0)}%, and 3rd class at {survival_by_class.get(3, 0)}%."
)
print(
    f"3. Across age bands, the highest survival rate was in the "
    f"{survival_by_age.index[0]} group at {survival_by_age.iloc[0]}%."
)
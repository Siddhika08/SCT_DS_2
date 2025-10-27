# 🚢 Titanic Dataset - Exploratory Data Analysis (EDA)
# Author: Siddhika Pandey

# ----------------------------
# STEP 1: Import Libraries
# ----------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# For clean, attractive visuals
sns.set(style="whitegrid")

# ----------------------------
# STEP 2: Load Dataset
# ----------------------------
# Titanic dataset is built into Seaborn, so we can load it directly
df = sns.load_dataset('titanic')

# Display first 5 rows
print("🔹 Preview of Dataset:")
print(df.head(), "\n")

# ----------------------------
# STEP 3: Data Cleaning
# ----------------------------
print("🔹 Checking for missing values:\n", df.isnull().sum())

# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['deck'].fillna('Unknown', inplace=True)

# Drop rows with missing fare values
df.dropna(subset=['fare'], inplace=True)

print("\n✅ Missing values handled successfully!\n")

# ----------------------------
# STEP 4: Basic Info
# ----------------------------
print("🔹 Dataset Info:")
print(df.info(), "\n")

print("🔹 Summary Statistics:")
print(df.describe(), "\n")

# ----------------------------
# STEP 5: Exploratory Data Analysis (Visuals)
# ----------------------------

# 1️⃣ Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='survived', data=df, palette='coolwarm')
plt.title("Survival Count (0 = Not Survived, 1 = Survived)")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.savefig("output_survival_count.png", dpi=300)
plt.show()

# 2️⃣ Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='survived', data=df, palette='Set2')
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.savefig("output_survival_by_gender.png", dpi=300)
plt.show()

# 3️⃣ Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x='pclass', hue='survived', data=df, palette='muted')
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.savefig("output_survival_by_class.png", dpi=300)
plt.show()

# 4️⃣ Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("output_age_distribution.png", dpi=300)
plt.show()

# 5️⃣ Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='YlGnBu')
plt.title("Correlation Heatmap")
plt.savefig("output_correlation_heatmap.png", dpi=300)
plt.show()

# ----------------------------
# STEP 6: Insights
# ----------------------------
print("""
🔍 Key Insights:
1️⃣ About 38% of passengers survived.
2️⃣ Females had a much higher survival rate than males.
3️⃣ 1st class passengers were more likely to survive.
4️⃣ Children had slightly higher survival chances than adults.
5️⃣ Higher fare passengers (usually 1st class) had better survival odds.

✅ Analysis completed successfully!
""")

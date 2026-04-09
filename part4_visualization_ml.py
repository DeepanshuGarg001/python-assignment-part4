import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# Task 1 — Data Exploration with Pandas
# ==========================================
print("--- Task 1: Data Exploration ---")

# 1. Load the dataset
df = pd.read_csv("students.csv")

# 2. Print the first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# 3. Print shape and dtypes
print(f"\nShape: {df.shape}")
print("\nData Types:")
print(df.dtypes)

# 4. Print summary statistics
print("\nSummary Statistics:")
print(df.describe())

# 5. Print count of passed/failed
print("\nPassed/Failed Counts:")
print(df['passed'].value_counts())

# 6. Compute and print average score per subject for passing/failing students
subject_cols = ['math', 'science', 'english', 'history', 'pe']
print("\nAverage scores for Students who PASSED:")
print(df[df['passed'] == 1][subject_cols].mean())

print("\nAverage scores for Students who FAILED:")
print(df[df['passed'] == 0][subject_cols].mean())

# 7. Find and print student with highest overall average
df['overall_avg'] = df[subject_cols].mean(axis=1)
top_student = df.loc[df['overall_avg'].idxmax()]
print(f"\nStudent with highest overall average: {top_student['name']} ({top_student['overall_avg']:.2f})")

# ==========================================
# Task 2 — Data Visualization with Matplotlib
# ==========================================
print("\n--- Task 2: Data Visualization with Matplotlib ---")

# Ensure avg_score is present
df['avg_score'] = df[subject_cols].mean(axis=1)

# 1. Bar Chart — Average score per subject across all students
plt.figure(figsize=(8, 5))
plt.bar(subject_cols, df[subject_cols].mean(), color='skyblue')
plt.title('Average Score per Subject')
plt.xlabel('Subject')
plt.ylabel('Average Score')
plt.savefig('plot1_bar.png')
print("Saved plot1_bar.png")

# 2. Histogram — Distribution of math scores
plt.figure(figsize=(8, 5))
plt.hist(df['math'], bins=5, color='coral', edgecolor='black')
plt.axvline(df['math'].mean(), color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {df["math"].mean():.2f}')
plt.title('Distribution of Math Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('plot2_hist.png')
print("Saved plot2_hist.png")

# 3. Scatter Plot — study_hours_per_day vs avg_score
plt.figure(figsize=(8, 5))
plt.scatter(df[df['passed']==1]['study_hours_per_day'], df[df['passed']==1]['avg_score'], color='green', label='Pass')
plt.scatter(df[df['passed']==0]['study_hours_per_day'], df[df['passed']==0]['avg_score'], color='red', label='Fail')
plt.title('Study Hours vs Average Score')
plt.xlabel('Study Hours per Day')
plt.ylabel('Average Score')
plt.legend()
plt.savefig('plot3_scatter.png')
print("Saved plot3_scatter.png")

# 4. Box Plot — attendance_pct for passing vs failing
plt.figure(figsize=(8, 5))
pass_attendance = df[df['passed']==1]['attendance_pct'].tolist()
fail_attendance = df[df['passed']==0]['attendance_pct'].tolist()
plt.boxplot([pass_attendance, fail_attendance], labels=['Pass', 'Fail'])
plt.title('Attendance Percentage: Pass vs Fail')
plt.xlabel('Status')
plt.ylabel('Attendance %')
plt.savefig('plot4_box.png')
print("Saved plot4_box.png")

# 5. Line Plot — math and science scores per student
plt.figure(figsize=(10, 6))
plt.plot(df['name'], df['math'], marker='o', linestyle='-', label='Math')
plt.plot(df['name'], df['science'], marker='s', linestyle='--', label='Science')
plt.title('Math and Science Scores per Student')
plt.xlabel('Student Name')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('plot5_line.png')
print("Saved plot5_line.png")

# ==========================================
# Task 3 — Data Visualization with Seaborn
# ==========================================
print("\n--- Task 3: Data Visualization with Seaborn ---")

# 1. Seaborn bar plot split by passed
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(data=df, x='passed', y='math', hue='passed', palette='muted', legend=False)
plt.title('Avg Math Score by Pass/Fail')
plt.subplot(1, 2, 2)
sns.barplot(data=df, x='passed', y='science', hue='passed', palette='muted', legend=False)
plt.title('Avg Science Score by Pass/Fail')
plt.savefig('plot6_seaborn_bar.png')
print("Saved plot6_seaborn_bar.png")

# 2. Seaborn scatter plot with regression
plt.figure(figsize=(8, 5))
sns.regplot(data=df[df['passed']==1], x='attendance_pct', y='avg_score', label='Pass', color='green')
sns.regplot(data=df[df['passed']==0], x='attendance_pct', y='avg_score', label='Fail', color='red')
plt.title('Attendance vs Average Score (with Regression)')
plt.legend()
plt.savefig('plot7_seaborn_scatter.png')
print("Saved plot7_seaborn_scatter.png")

# 3. Comparison Comment
# Seaborn vs Matplotlib:
# Seaborn provides higher-level abstractions for statistical plotting (e.g., automatic mean calculation in bar plots)
# and better default aesthetics. Matplotlib provides lower-level control which is useful for custom layouts.
# For these tasks, Seaborn made handling grouped data and regression lines much simpler.

# ==========================================
# Task 4 — Machine Learning with scikit-learn
# ==========================================
print("\n--- Task 4: Machine Learning with scikit-learn ---")

# Step 1 — Prepare Data
features = ['math', 'science', 'english', 'history', 'pe', 'attendance_pct', 'study_hours_per_day']
X = df[features]
y = df['passed']

# Split into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2 — Train a Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2%}")

# Step 3 — Evaluate the Model
y_pred = model.predict(X_test_scaled)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")

print("\nIndividual Predictions for Test Set:")
for idx, (actual, pred) in enumerate(zip(y_test, y_pred)):
    student_name = df.loc[X_test.index[idx], 'name']
    status = "✅ correct" if actual == pred else "❌ wrong"
    print(f"{student_name}: Actual={actual}, Predicted={pred} — {status}")

# Step 4 — Feature Importance
coefs = model.coef_[0]
feature_importance = sorted(zip(features, coefs), key=lambda x: abs(x[1]), reverse=True)

print("\nFeature Importance (Sorted by Absolute Value):")
for feat, coef in feature_importance:
    print(f"{feat}: {coef:.4f}")

# Horizontal bar chart for feature importance
plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in [x[1] for x in feature_importance]]
plt.barh([x[0] for x in feature_importance], [x[1] for x in feature_importance], color=colors)
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.xlabel('Coefficient Value (Positive pushes towards Pass, Negative towards Fail)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show() # This will only show if in an interactive environment, but good practice.

# Step 5 — Predict for a New Student (Bonus)
new_student_data = [[75, 70, 68, 65, 80, 82, 3.2]]
new_student_scaled = scaler.transform(new_student_data)
prediction = model.predict(new_student_scaled)[0]
prob = model.predict_proba(new_student_scaled)[0]

print(f"\nNew Student Prediction: {'Pass' if prediction == 1 else 'Fail'}")
print(f"Probability: Pass={prob[1]:.2%}, Fail={prob[0]:.2%}")

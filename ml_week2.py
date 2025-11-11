# ml_week2_complete.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("=== WEEK 2: COMPLETE ML PIPELINE ===")

# Load data
df = pd.read_csv('final_analysis_results.csv')

# Create target and features
median_threshold = df['stations_per_million'].median()
df['infrastructure_class'] = df['stations_per_million'].apply(
    lambda x: 'Good' if x > median_threshold else 'Poor'
)

# Feature engineering
df['population_density_feature'] = df['2022 Population'] / 1000000
df['stations_per_population'] = df['station_count'] / df['2022 Population']

features = ['population_density_feature', 'station_count', 'stations_per_population']
X = df[features]
y = df['infrastructure_class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 1. MODEL COMPARISON
print("\n🔍 MODEL COMPARISON:")
models = {
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'SVM': SVC(random_state=42, class_weight='balanced')
}

best_score = 0
best_model_name = ""

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Train and test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {accuracy:.3f}")

    if accuracy > best_score:
        best_score = accuracy
        best_model_name = name
        best_model = model

print(f"\n🏆 BEST MODEL: {best_model_name} (Accuracy: {best_score:.3f})")

# 2. HYPERPARAMETER TUNING (for Random Forest)
print("\n⚙️ HYPERPARAMETER TUNING...")
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

    # Use best model
    best_model = grid_search.best_estimator_

# 3. FINAL MODEL EVALUATION
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 FINAL MODEL PERFORMANCE:")
print(f"Model: {best_model_name}")
print(f"Accuracy: {final_accuracy:.2f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    print(f"\n🔍 FEATURE IMPORTANCE:")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)

# 4. SAVE MODEL
joblib.dump(best_model, 'best_ev_model.pkl')
print(f"\n💾 Model saved as 'best_ev_model.pkl'")

# 5. BUSINESS RECOMMENDATIONS
print(f"\n🎯 BUSINESS RECOMMENDATIONS:")

# Predict on all data
df['predicted_class'] = best_model.predict(X)
df['prediction_probability'] = best_model.predict_proba(X)[:, 1]

# High priority countries (large population + poor infrastructure)
high_priority = df[
    (df['predicted_class'] == 'Poor') &
    (df['2022 Population'] > 30000000)  # Countries with >30M people
    ].sort_values('stations_per_million')

print("High-Priority Countries for Investment:")
for _, country in high_priority.iterrows():
    print(f"  • {country['Country/Territory']}: {country['stations_per_million']:.2f} stations/million")

# Success stories
success_stories = df[
    (df['predicted_class'] == 'Good') &
    (df['stations_per_million'] > 10)
    ].sort_values('stations_per_million', ascending=False)

print(f"\nSuccess Stories (Learn From These):")
for _, country in success_stories.head(3).iterrows():
    print(f"  • {country['Country/Territory']}: {country['stations_per_million']:.1f} stations/million")

# Save final results
df.to_csv('week2_complete_analysis.csv', index=False)
print(f"\n✅ Complete analysis saved to 'week2_complete_analysis.csv'")
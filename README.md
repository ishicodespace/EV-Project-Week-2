# EV Charging Infrastructure Analysis

## Week 1: Data Analysis & Week 2: Machine Learning Model

### Problem Statement
Identify countries with poor EV charging infrastructure to guide sustainable transportation investments using data analysis and machine learning.

### Week 1 - Data Analysis Results
- Analyzed 9,159 charging stations across 67 countries
- **Top Performers**: Canada (107.5 stations/million), Liechtenstein (101.7), Denmark (50.3)
- **Critical Gaps**: China (0.00), India (0.02), Indonesia (0.03)
- **Global Average**: 5.77 stations/million people

### Week 2 - Machine Learning Model
Built a classification model to predict infrastructure quality with **95% accuracy**:

**Key Improvisations:**
- Balanced data using median threshold (0.7) instead of mean
- Created "stations per person" feature (58% importance)
- Compared multiple algorithms (Random Forest, Logistic Regression, SVM)
- Fine-tuned hyperparameters for optimal performance
- Generated actionable investment priorities

**Model Performance:**
- Accuracy: 95%
- Precision: 96%
- Recall: 95%
- Only 1 misclassification (Tunisia)

### Business Impact
The model identifies high-priority countries for EV infrastructure investment, supporting sustainable transportation planning and resource allocation.

### Files
- `data_cleaning.py` - Data preprocessing pipeline
- `ml_week2.py` - Full ML implementation
- `best_ev_model.pkl` - Trained model for deployment
- `week2_complete_analysis.csv` - Final results with predictions

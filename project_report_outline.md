# IBM–NASSCOM Project Report Outline

## 1. Title
Fake News Detection Using Machine Learning

## 2. Abstract
Develop a supervised NLP classification system that predicts whether a news article belongs to the fake or real class using TF-IDF features and classical machine-learning classifiers.

## 3. Problem Statement
The rapid spread of online misinformation makes manual screening difficult. An automated first-pass classifier can help identify suspicious textual patterns for further human verification.

## 4. Objectives
- Collect a labeled fake/real news dataset.
- Clean and preprocess textual data.
- Extract TF-IDF features.
- Train multiple ML classifiers.
- Compare models using standard classification metrics.
- Deploy the selected model through a simple web interface.

## 5. Methodology
Dataset → preprocessing → train/test split → TF-IDF → model training → evaluation → best-model selection → deployment.

## 6. Models
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

## 7. Evaluation
Use the actual values generated in `results/model_comparison.csv`.
Include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## 8. Limitations
- Dataset/domain bias
- News styles evolve over time
- Text-only classification cannot verify facts independently
- Predictions should support, not replace, human fact-checking

## 9. Future Scope
- Transformer/BERT-based models
- Multilingual detection
- Source credibility features
- Retrieval-based evidence verification
- Explainable AI

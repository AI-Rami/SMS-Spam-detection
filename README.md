
✉️ SMS Spam Detection with Machine Learning

This project is a machine learning classifier that detects whether a text message (SMS) is spam or ham (not spam). It was developed as part of a school project and uses standard natural language processing (NLP) and classification techniques to demonstrate the process of building a real-world text classifier.

---

Overview

- Dataset: spam.csv (contains 5,572 SMS messages labeled as "spam" or "ham")
- Target column: label
- Input column: message
- Models used: Logistic Regression, SVM, Naive Bayes
- Imbalance handling: discussed but not applied in final version

---

Features

- Data cleaning and preprocessing
- Text vectorization using TF-IDF and Word2Vec
- Multiple model comparison (accuracy, F1-score, ROC-AUC)
- Easy-to-understand code and outputs
- Deployment using Streamlit

---

Model Performance (Example)

| Model              | Accuracy | F1 Score | ROC-AUC |
|--------------------|----------|----------|---------|
| Logistic Regression| 97%      | 0.96     | 0.98    |
| SVM                | 96%      | 0.95     | 0.97    |
| Naive Bayes        | 94%      | 0.92     | 0.96    |

Note: These are example scores. Actual results may vary slightly depending on the text representation method.

---

Tech Stack

- Python
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- NLTK / Word2Vec (optional)
- Jupyter Notebook

---

How to Run

1. Clone the repo:
   git clone https://github.com/YOUR_USERNAME/sms-spam-detector.git
   cd sms-spam-detector

2. Install dependencies:
   pip install -r requirements.txt

3. Open the notebook:
   jupyter notebook task_3_Spam_detection.ipynb

---

License

This project is licensed under the MIT License.

---

Acknowledgments

- SMS Spam Collection dataset from UCI Machine Learning Repository
- Developed as part of a school assignment



# 🧠 Disease Detection System using Machine Learning

This project is an intelligent disease detection system that predicts a possible illness based on a user's selected symptoms. It uses machine learning (Naive Bayes + SMOTE + feature selection) and is deployed using **Streamlit**.


---

## 🚀 Live Demo

👉 [Click here to use the app](https://your-deployed-app-url.streamlit.app) <!-- Replace this with actual Streamlit app URL -->

---



## ✅ Features

- 🧠 Predicts disease based on binary symptom inputs  
- ⚖️ Uses **SMOTE** for class balancing  
- 🧪 Evaluates models with validation + test set  
- 🧮 Ensemble model with **VotingClassifier** (Naive Bayes + Random Forest)  
- 📊 Confusion matrix + feature importance  
- 🌐 Deployed on **Streamlit Cloud**

---

## 📊 ML Models Compared

- Naive Bayes ✅ (Best performing)
- Decision Tree
- Random Forest
- Ensemble: VotingClassifier (Soft voting with NB + RF)

---

## 🛠️ Tech Stack

- Python, Pandas, Scikit-learn
- imbalanced-learn (SMOTE)
- Streamlit
- Matplotlib + Seaborn
- Git, GitHub

---

## 🧪 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Krushbiradar18/disease-detection-ml.git
cd disease-detection-ml

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python main.py

# Launch the app
streamlit run app.py


⸻

📈 Sample Results

Metric	Score
Accuracy	67.92%
Best Model	Naive Bayes


⸻

🤖 Future Improvements
	•	Add symptom severity or duration
	•	Integrate with clinical datasets
	•	Deploy using Docker or HuggingFace Spaces
	•	Add REST API for external use

⸻

👩‍💻 Developed by

Krushnali Biradar


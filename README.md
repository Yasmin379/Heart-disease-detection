# ❤️ Heart Disease Detection Using Machine Learning

## 📌 Project Overview

Heart Disease Detection is a machine learning project developed to predict whether a person is likely to have heart disease based on medical attributes.

The project follows a complete machine learning workflow:

**Data → Preprocessing → EDA → Model Training → Hyperparameter Tuning → Evaluation → Prediction**

Multiple machine learning algorithms were implemented and compared to identify a suitable model for heart disease prediction.

---

## 🎯 Objectives

- Analyze and understand the heart disease dataset.
- Perform exploratory data analysis (EDA).
- Preprocess the dataset for machine learning.
- Train multiple classification models.
- Tune model hyperparameters using GridSearchCV.
- Evaluate models using different performance metrics.
- Compare model performance.
- Deploy the trained models through a Streamlit application.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Machine learning
- **Joblib / Pickle** – Model saving and loading
- **Streamlit** – Web application
- **Git & GitHub** – Version control

---

## 📂 Project Structure

```text
Heart-disease-detection/
│
├── data/
│   └── heart_disease.csv
│
├── models/
│   ├── decision_tree.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── results/
│   └── confusion_matrices/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── app.py
├── main.py
├── save_models.py
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md


📊 Dataset

The project uses a heart disease dataset containing medical attributes of patients.

The target variable is:

heart_disease

The target represents whether the patient has heart disease.

The dataset is used for both exploratory analysis and supervised machine learning classification



🔄 Project Workflow
             Dataset
                ↓
       Data Preprocessing
                ↓
              EDA
                ↓
        Train-Test Split
                ↓
        Model Training
                ↓
      Hyperparameter Tuning
                ↓
         Model Evaluation
                ↓
       Model Comparison
                ↓
       Streamlit Application



🔍 Exploratory Data Analysis

EDA was performed to understand the dataset before training the models.

The analysis includes:

Dataset structure
Feature distributions
Target variable distribution
Missing-value analysis
Correlation analysis
Visualization of important features

The results of EDA are available in:

notebooks/EDA.ipynb



⚙️ Data Preprocessing

The preprocessing stage includes:

Loading the dataset.
Separating features and target variable.
Splitting the dataset into training and testing sets.
Using stratified splitting to maintain the target distribution.
Applying feature scaling where required by the model.

The preprocessing functionality is implemented in:

src/preprocessing.py



🤖 Machine Learning Models

The project uses four machine learning classification algorithms:

1. Logistic Regression

A simple and widely used classification algorithm used as one of the baseline models.

2. Decision Tree

A tree-based classification algorithm that makes decisions using feature-based splits.

3. Random Forest

An ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

4. Support Vector Machine (SVM)

A classification algorithm that finds a suitable decision boundary between different classes.




🔧 Hyperparameter Tuning

GridSearchCV was used to tune model hyperparameters.

GridSearchCV tests different combinations of predefined hyperparameters using cross-validation and selects the combination that performs best.

For example, Random Forest parameters included:

n_estimators
max_depth
min_samples_split

For SVM, parameters included:

C
gamma
kernel



📈 Model Evaluation

The trained models were evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix

These metrics provide different perspectives on model performance.

Why these metrics?

Accuracy
Measures the overall percentage of correct predictions.

Precision
Measures how many predicted positive cases were actually positive.

Recall
Measures how many actual positive cases were correctly identified.

F1-Score
Provides a balance between precision and recall.

Confusion Matrix
Shows the number of correct and incorrect predictions for each class.



🌲 Random Forest Results

The best Random Forest parameters obtained using GridSearchCV were:

Parameter	Value
n_estimators	50
max_depth	5
min_samples_split	5
Performance
Metric	Score
Accuracy	71.25%
Precision	69.81%
Recall	84.09%
F1-Score	76.29%



📐 SVM Results

The best SVM parameters obtained using GridSearchCV were:

Parameter	Value
C	0.1
gamma	scale
kernel	linear
Performance
Metric	Score
Accuracy	66.25%
Precision	68.89%
Recall	70.45%
F1-Score	69.66%



📊 Model Comparison
Model	Accuracy	Precision	Recall	F1-Score
Random Forest	71.25%	69.81%	84.09%	76.29%
SVM	66.25%	68.89%	70.45%	69.66%

The project also includes Logistic Regression and Decision Tree models for comparison.

The final comparison can be viewed through the model evaluation results and visualizations.



🖥️ Streamlit Application

The trained machine learning models are integrated into a Streamlit-based application.

The application allows users to:

Enter the required patient information.
Submit the information.
Load the trained machine learning model.
Generate a heart disease prediction.
Display the prediction to the user.

The main application is:

app.py



🚀 How to Run the Project
1. Clone the Repository
git clone <repository-url>

Move into the project directory:

cd Heart-disease-detection

2. Create Virtual Environment
Windows
python -m venv env

Activate it:

env\Scripts\activate

3. Install Dependencies

Install the required packages:

pip install -r requirements.txt
4. Run the Streamlit Application
streamlit run app.py

The application will open in your browser.




🌿 Git & GitHub Guidelines

To keep the project organized, team members should work on separate branches.

Before starting work
git checkout main
git pull origin main
Create your own branch
git checkout -b your-name

Check your changes
git status
Add changes
git add .
Commit changes
git commit -m "Describe your changes"

Example:

git commit -m "Add model evaluation"
Push your branch
git push origin your-name

After pushing, create a Pull Request on GitHub.



⚠️ Important Git Rules
Do not directly push to main.
Always create and work on your own branch.
Pull the latest changes from main before starting new work.
Create a Pull Request after completing your work.
Do not commit the virtual environment.
Do not modify another teammate's work unnecessarily.
Resolve merge conflicts carefully before merging.
Always check git status before pushing.



👥 Team Contributions
Team Member	Contribution
Himanshu Salunkhe 	Data loading, preprocessing and EDA
kajal Dinde        	Machine learning model development
Yasmin Deshmukh   	Model training, evaluation and visualization
Sarthak Shitole   	Application integration



📁 Important Files
File / Folder	Purpose
data/	Contains the dataset
notebooks/	Contains EDA and experimentation notebooks
src/preprocessing.py	Data preprocessing
src/evaluate.py	Model evaluation and visualization
models/	Saved trained ML models
results/	Evaluation results and graphs
app.py	Streamlit application
requirements.txt	Python dependencies
README.md	Project documentation



⚠️ Disclaimer

This project is developed for educational and academic purposes.

The predictions generated by the application should not be considered a medical diagnosis. A qualified healthcare professional should always be consulted for medical decisions.

🙌 Conclusion

The Heart Disease Detection project demonstrates the complete machine learning workflow, starting from data analysis and preprocessing to model training, evaluation, and application development.

Multiple classification models were trained and compared, and the trained models were integrated into a user-friendly Streamlit application for heart disease prediction.

👨‍💻 Team

Heart Disease Detection – Machine Learning Project

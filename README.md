🏠 House Price Prediction using Machine Learning
📌 Project Overview
This project aims to accurately predict house prices using regression-based machine learning models.
We used structured data containing various features like area, number of bedrooms, bathrooms, furnishing status, and more.

The project was built using:

Python for data processing and modeling

Streamlit for building a responsive web application

Scikit-learn and XGBoost for modeling

This project follows the Agile SDLC approach, progressing through planning, data prep, modeling, evaluation, deployment, and maintenance phases.

📁 Dataset Details
Feature	Description
Source	Kaggle - Housing Price Dataset
Total Records	545
Target Variable	price (INR)
Feature Types	Numerical + Categorical
Key Features	area, bedrooms, bathrooms, AC, parking, furnishingstatus, etc.

🧠 ML Pipeline Overview
Data Loading: Loaded dataset from CSV using Pandas.

Preprocessing:

Binary encoding for yes/no features

One-hot encoding for furnishingstatus

Train-test split (80-20)

Model Training:

Trained 3 models: Linear Regression, Random Forest, and XGBoost

Evaluation:

Used MAE, MSE, and R² Score

Best Model:

Random Forest selected based on highest R²

Deployment:

Web app built in Streamlit to input features and predict price live

Model Storage:

Trained model saved using joblib to /models/final_model.pkl

⚙️ Tech Stack
Tool	Purpose
Python	Core programming language
Pandas & NumPy	Data handling
Scikit-learn	Model training & evaluation
XGBoost	Gradient boosting model
Matplotlib & Seaborn	EDA and visualizations
Streamlit	Web app for UI
Joblib	Saving trained model

🤖 Models Used & Comparison
Model	MAE (↓)	MSE (↓)	R² Score (↑)
Linear Regression	6.5 L	1.2 Cr	0.82
Random Forest ✅	4.3 L	0.7 Cr	0.87
XGBoost Regressor	5.1 L	0.9 Cr	0.85

✅ Random Forest was selected due to the best overall performance.

🚀 Streamlit Web App
The trained model is deployed using Streamlit, allowing users to:

Enter property details (area, bedrooms, bathrooms, etc.)

Click "Predict"

Instantly view the predicted house price

🖥️ How to Run This Project
1. Clone or Download
bash
Copy
Edit
git clone <repo-url>
cd house_price_prediction
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the web app
bash
Copy
Edit
streamlit run app/app.py
📦 Project Structure
cpp
Copy
Edit
house_price_prediction/
├── app/
│   └── app.py
├── data/
│   └── Housing.csv
├── models/
│   └── final_model.pkl
├── notebook/
│   └── EDA.ipynb (optional)
├── main.py
├── requirements.txt
├── README.md
✅ Features Completed
 Data Cleaning & Preprocessing

 Model Training & Evaluation

 Saving Trained Model

 Streamlit Deployment

 Custom UI Styling

 PDF Report + PPT Presentation

 Viva-ready project structure

📌 Author
👨‍💻 shubham.
Solo Developer – student.
Tools: Python, VS Code, Streamlit, Kaggle, Git














To

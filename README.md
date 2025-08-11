Heart Sight – Heart Attack Prediction using Machine Learning

📌 Overview -
Heart Sight is a machine learning project that predicts the likelihood of a heart attack using two powerful algorithms — AdaBoost and Histogram-based Gradient Boosting.
The application is built with Streamlit and runs locally in a web browser, providing an interactive interface for quick and easy predictions.

🎯 Project Objective -
To develop a predictive model that can help in the early detection of heart attack risk using clinical parameters, enabling proactive medical consultation.

🚀 Features -
Dual Algorithm Approach:
  AdaBoost → Accuracy: 81%
  HistGradientBoosting → Accuracy: 86%
Interactive Web Interface with Streamlit
User-friendly Inputs for entering medical details
Instant Risk Prediction displayed in browser
Model Comparison for performance evaluation

🛠 Tech Stack -
Python 3.x
Machine Learning: scikit-learn (AdaBoost, HistGradientBoosting)
Data Processing: pandas, numpy
Visualization: matplotlib, seaborn
Web App: Streamlit

📂 Project Structure
HeartSight/

├── app.py                     # Streamlit application

├── heartattack.ipynb          # Model training & evaluation notebook

├── requirements.txt           # Python dependencies

├── dataset.csv                # Heart disease dataset

└── README.md                  # Project documentation


📦 Installation & Usage

1️⃣ Clone the Repository
git clone https://github.com/YOUR-USERNAME/HeartSight.git
cd HeartSight

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run app.py

4️⃣ Open in Browser
Streamlit will automatically open your default browser at:
http://localhost:8501

📊 Model Performance
Algorithm	Accuracy
AdaBoost	81%
HistGradientBoosting	86%

🌐 Deployment
Currently deployed locally via Streamlit.
Can be deployed on Streamlit Cloud, Heroku, or similar platforms for online access.

📸 Screenshots
1.Home Page
![WhatsApp Image 2025-08-09 at 17 35 18_95c0abed](https://github.com/user-attachments/assets/3678aa7d-893a-49ec-ab35-bb9d255f2a54)

2.Data Pre-Processing 
![WhatsApp Image 2025-08-09 at 17 35 18_90acc93d](https://github.com/user-attachments/assets/c8b63926-3ce8-4f5d-8d53-e58d7b993065)

3.Graphs

i) Age distribution
![WhatsApp Image 2025-08-09 at 17 35 18_4a2c2267](https://github.com/user-attachments/assets/56792992-bd11-4c70-9941-7f2baf41ec72)

ii) Feature Correlation
![WhatsApp Image 2025-08-09 at 17 35 18_91e1af05](https://github.com/user-attachments/assets/acfb8cf5-7f9d-479d-b4f9-a0de0b12b3f1)

iii) Pairplot
![WhatsApp Image 2025-08-09 at 17 35 19_8fc63e69](https://github.com/user-attachments/assets/d02299b1-fad9-4489-acb6-467a378b75a4)

4.Model Training

i) Hist Gradient Algorithm
![WhatsApp Image 2025-08-09 at 17 35 19_5d42e2ff](https://github.com/user-attachments/assets/764a63ef-ac96-46fa-9c52-b41ac7eda2ba)

ii) Ada Boost Algorithm
![WhatsApp Image 2025-08-09 at 17 35 19_0d3a6bb7](https://github.com/user-attachments/assets/83b5d853-b4cd-472d-9e92-751f04e84848)

5.Model Evaluation

i) Hist Gradient Algorithm
![WhatsApp Image 2025-08-09 at 17 35 19_6ebb7558](https://github.com/user-attachments/assets/49113594-fd04-4f58-8d80-68ec300ad48a)

ii) Ada Boost Algorithm
![WhatsApp Image 2025-08-09 at 17 50 03_0d955ecc](https://github.com/user-attachments/assets/f23acfa5-2f9b-4410-9e03-23c218f51c61)












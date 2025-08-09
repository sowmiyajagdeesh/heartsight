import streamlit as st
import pandas as pd
import pickle
import bz2
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier


# Define models and hyperparameters
MODELS = {
    "AdaBoost": (AdaBoostClassifier(), {
    'n_estimators': [10, 20, 30, 40, 50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0, 10.0]
    }),
        "Hist Gradient Boosting": (HistGradientBoostingClassifier(random_state=42), {
        'max_iter': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
    })
}
st.set_page_config(layout="wide")
# Function to load data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Train the model
def train_model(model, param_grid, X_train, y_train):
    st.write("### Training Model...")
    progress_bar = st.progress(0)
    for i in range(5):  # Simulate a progress bar with a 5-second delay
        time.sleep(1)
        progress_bar.progress((i + 1) * 20)
    clf = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0
    )
    clf.fit(X_train, y_train)
    progress_bar.empty()
    return clf

# Save model to a compressed file
def save_model_to_file(model, file_name):
    try:
        with bz2.BZ2File(file_name, "wb") as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        st.error(f"Error saving the model: {e}")
        return False

# Load model from a compressed file
def load_model_from_file(uploaded_file):
    try:
        with bz2.BZ2File(uploaded_file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Initialize session state variables
if "X_train" not in st.session_state:
    st.session_state["X_train"] = None
if "X_test" not in st.session_state:
    st.session_state["X_test"] = None
if "y_train" not in st.session_state:
    st.session_state["y_train"] = None
if "y_test" not in st.session_state:
    st.session_state["y_test"] = None
if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None
if "best_params" not in st.session_state:
    st.session_state["best_params"] = None

# Sidebar navigation
st.sidebar.title("Navigation")
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

if st.sidebar.button("Home"):
    st.session_state["current_page"] = "Home"
if st.sidebar.button("Dataset & Processing"):
    st.session_state["current_page"] = "Dataset & Processing"
if st.sidebar.button("Model Training"):
    st.session_state["current_page"] = "Model Training"
if st.sidebar.button("Model Evaluation"):
    st.session_state["current_page"] = "Model Evaluation"

# Home Page
if st.session_state["current_page"] == "Home":
    # Use Streamlit's image handling to ensure visibility
    from PIL import Image  # Import for handling images
    st.markdown(
        """
        <style>
        .content {
            text-align: left;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Load the image (local file or replace with an online URL)
    try:
        # If using a local image, ensure 'heart.jpeg' exists in the working directory
        image = Image.open("heart.jpeg")  # Replace with your image file path
    except FileNotFoundError:
        # Fallback to an online image
        image_url = "https://source.unsplash.com/1920x1080/?heart,health"
        st.image(image_url, caption="Heart Health", use_column_width=True)
    else:
        st.image(image, use_column_width=True)

    # Add the title and content below the image
    st.markdown(
        """
        <div class="content">
            <h1>Heart Disease Prediction Using Machine Learning</h1>
            <p>This app provides an end-to-end solution for building a machine learning model.</p>
            <p>Here’s what you can do:</p>
            <ul>
                <li>Upload and process datasets.</li>
                <li>Train models using different algorithms and tune hyperparameters.</li>
                <li>Evaluate trained models and view performance metrics.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# Page: Dataset & Processing
elif st.session_state["current_page"] == "Dataset & Processing":
    st.title("Dataset and Data Processing")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        data = load_data(uploaded_file)
        st.write(f"Dataset contains **{data.shape[0]} rows** and **{data.shape[1]} columns**")
        st.dataframe(data.head())
        X = data.drop(columns=["output"])  # Assuming "output" is the target
        y = data["output"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        
                        # Display graphs
        st.write("### Dataset Insights")

        # Age Distribution
        st.write("#### Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data["age"], kde=True, ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)

        # Correlation Heatmap
        st.write("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation")
        st.pyplot(fig)

        # Pairplot
        st.write("#### Pairplot of Selected Features")
        selected_features = ["age", "chol", "thalachh", "output"]
        pairplot_fig = sns.pairplot(data[selected_features], hue="output", diag_kind="kde")
        st.pyplot(pairplot_fig.fig)

        st.success("Dataset processed successfully!")

    if st.button("Go to Model Training"):
        st.session_state["current_page"] = "Model Training"
        st.experimental_rerun()

# Page: Model Training
elif st.session_state["current_page"] == "Model Training":
    st.title("Model Training")
    model_name = st.selectbox("Select a model", options=list(MODELS.keys()))
    if model_name:
        model, param_grid = MODELS[model_name]
        if st.button("Train Model"):
            clf = train_model(model, param_grid, st.session_state["X_train"], st.session_state["y_train"])
            st.session_state["trained_model"] = clf.best_estimator_
            st.session_state["best_params"] = clf.best_params_
            st.success("Model trained successfully!")

    if st.session_state["trained_model"]:
        st.write("### Best Parameters")
        st.json(st.session_state["best_params"])

        save_model_name = st.text_input("Enter a name to save the model (optional):")
        if st.button("Save Model"):
            if save_model_name:
                file_name = f"{save_model_name}.pkl.bz2"
                if save_model_to_file(st.session_state["trained_model"], file_name):
                    st.success(f"Model saved successfully as {file_name}!")

    if st.button("Go to Model Evaluation"):
        st.session_state["current_page"] = "Model Evaluation"
        st.experimental_rerun()

# Page: Model Evaluation
elif st.session_state["current_page"] == "Model Evaluation":
    st.title("Model Evaluation")
    uploaded_model = st.file_uploader("Upload a pre-trained model (optional)", type=["bz2"])
    if uploaded_model:
        model = load_model_from_file(uploaded_model)
        if model:
            st.session_state["trained_model"] = model
            st.success("Uploaded model loaded successfully!")

    if st.session_state["trained_model"]:
        if st.button("Evaluate Model"):
            st.write("### Evaluating Model...")
            time.sleep(5)  # Add a 5-second delay
            y_pred = st.session_state["trained_model"].predict(st.session_state["X_test"])
            st.write("### Evaluation Metrics")
            st.text("Classification Report:")
            st.text(classification_report(st.session_state["y_test"], y_pred))
            st.write("**Accuracy:**", accuracy_score(st.session_state["y_test"], y_pred))

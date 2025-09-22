import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Load model & vectorizer ---
model = joblib.load("models/fake_job_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# --- Load dataset for visualization ---
df = pd.read_csv("data/fake_job_postings.csv")
y = df['fraudulent']

# --- Title ---
st.title("🕵️ Fake Job Detection System")
st.markdown("""
Detect fake job postings using machine learning.
You can enter a job description manually or upload a CSV file for bulk predictions.
""")

# --- Text cleaning function ---
def clean_text(text):
    if not isinstance(text, str):
        text = ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

# --- Single job prediction ---
st.subheader("Predict a Single Job")
job_desc = st.text_area("Enter the job description here:")

if st.button("Check Job"):
    if job_desc.strip() != "":
        job_vec = vectorizer.transform([clean_text(job_desc)])
        pred = model.predict(job_vec)[0]
        st.success("This job posting is " + ("Fake 🚫" if pred==1 else "Real ✅"))
    else:
        st.warning("Please enter a job description!")

# --- Bulk CSV upload ---
st.subheader("Bulk Prediction via CSV Upload")
uploaded_file = st.file_uploader("Upload CSV with a 'description' column", type=["csv"])
if uploaded_file:
    df_csv = pd.read_csv(uploaded_file)
    if 'description' in df_csv.columns:
        df_csv['description_clean'] = df_csv['description'].apply(clean_text)
        vec_csv = vectorizer.transform(df_csv['description_clean'])
        df_csv['prediction'] = model.predict(vec_csv)
        df_csv['prediction_label'] = df_csv['prediction'].apply(lambda x: "Fake 🚫" if x==1 else "Real ✅")
        st.dataframe(df_csv[['description','prediction_label']])
        st.success("✅ Bulk prediction completed!")
    else:
        st.error("CSV must contain a 'description' column.")

# --- Dataset insights ---
st.subheader("Dataset Insights")

# Distribution of Real vs Fake
fig1, ax1 = plt.subplots(figsize=(6,4))
sns.countplot(x=y, palette="viridis", ax=ax1)
ax1.set_xticklabels(["Real", "Fake"])
st.pyplot(fig1)

# Confusion matrix
real = df[df['fraudulent'] == 0]
fake = df[df['fraudulent'] == 1]
real_downsampled = real.sample(len(fake), random_state=42)
df_balanced = pd.concat([real_downsampled, fake]).reset_index(drop=True)
df_balanced['text'] = df_balanced['title'].fillna('') + ' ' + df_balanced['description'].fillna('') + ' ' + df_balanced['requirements'].fillna('')
df_balanced['text_clean'] = df_balanced['text'].apply(clean_text)
X_vec = vectorizer.transform(df_balanced['text_clean'])
X_train, X_test, y_train, y_test = train_test_split(X_vec, df_balanced['fraudulent'], test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real","Fake"], yticklabels=["Real","Fake"], ax=ax2)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

acc = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {acc:.2%}")

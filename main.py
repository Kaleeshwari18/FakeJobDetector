import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("data/fake_job_postings.csv")

real = df[df['fraudulent'] == 0]
fake = df[df['fraudulent'] == 1]
real_downsampled = real.sample(len(fake), random_state=42)
df_balanced = pd.concat([real_downsampled, fake]).reset_index(drop=True)

df_balanced['text'] = (
    df_balanced['title'].fillna('') + ' ' +
    df_balanced['description'].fillna('') + ' ' +
    df_balanced['requirements'].fillna('')
)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

df_balanced['text_clean'] = df_balanced['text'].apply(clean_text)

X = df_balanced['text_clean']
y = df_balanced['fraudulent']

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/fake_job_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("✅ Model and vectorizer saved successfully!")

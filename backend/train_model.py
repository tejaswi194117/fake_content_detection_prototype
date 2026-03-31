import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# PATH FIX
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(PROJECT_ROOT, "data/news.csv")

df = pd.read_csv(data_path)

# CLEAN DATA
df = df.dropna()
df = df[df['text'].str.strip() != ""]

X_text = df['text']
y = df['label'].astype(int)

# VECTORIZE
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X_text)

# MODEL
model = LogisticRegression()
model.fit(X, y)

# SAVE
os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/text_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("Model trained successfully")
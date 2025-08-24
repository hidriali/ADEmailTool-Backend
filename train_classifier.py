import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib

try:
    print("📥 Loading dataset...")
    df = pd.read_csv("random_emails_labeled.csv")
    print(f"✅ Loaded {len(df)} rows")

    X_raw = df["Email"].tolist()
    y = df["Label"].tolist()

    print("🔄 Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(X_raw, show_progress_bar=True)

    print("⚙️ Training classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    print("💾 Saving model...")
    joblib.dump(clf, "email_classifier.joblib")
    print("✅ Model saved as 'email_classifier.joblib'")

except Exception as e:
    print("❌ Error:", e)

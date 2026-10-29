
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import random
import threading

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# -----------------------------------------
# PROJECT PATHS AND SETTINGS
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "aclImdb"
MODEL_FILE = BASE_DIR / "sentiment_model.joblib"

TRAIN_PER_CLASS = 3000
TEST_PER_CLASS = 500
RANDOM_SEED = 42

model = None
accuracy = None


# -----------------------------------------
# LOAD A SAMPLE OF REVIEWS
# -----------------------------------------

def read_reviews(folder, limit):
    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{folder}\n\n"
            "Run download_dataset.py first."
        )

    files = list(folder.glob("*.txt"))

    if not files:
        raise ValueError(f"No review files found in:\n{folder}")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(files)
    files = files[:limit]

    reviews = []

    for file_path in files:
        try:
            review = file_path.read_text(
                encoding="utf-8",
                errors="replace"
            ).strip()

            if review:
                reviews.append(review)

        except OSError:
            continue

    if not reviews:
        raise ValueError(f"No readable reviews in:\n{folder}")

    return reviews


# -----------------------------------------
# TRAIN AND EVALUATE THE MODEL
# -----------------------------------------

def train_model():
    global model, accuracy

    try:
        status_label.config(
            text="Loading reviews and preparing the model..."
        )

        # If a saved model exists, reuse it.
        if MODEL_FILE.exists():
            status_label.config(
                text="Loading saved model..."
            )

            saved = joblib.load(MODEL_FILE)

            model = saved["model"]
            accuracy = saved["accuracy"]

            root.after(0, model_ready)
            return

        # Load a balanced training sample.
        train_positive = read_reviews(
            DATA_DIR / "train" / "pos",
            TRAIN_PER_CLASS
        )

        train_negative = read_reviews(
            DATA_DIR / "train" / "neg",
            TRAIN_PER_CLASS
        )

        # Load a separate sample for testing.
        test_positive = read_reviews(
            DATA_DIR / "test" / "pos",
            TEST_PER_CLASS
        )

        test_negative = read_reviews(
            DATA_DIR / "test" / "neg",
            TEST_PER_CLASS
        )

        X_train = train_positive + train_negative
        y_train = (
            ["positive"] * len(train_positive)
            + ["negative"] * len(train_negative)
        )

        X_test = test_positive + test_negative
        y_test = (
            ["positive"] * len(test_positive)
            + ["negative"] * len(test_negative)
        )

        status_label.config(
            text=f"Training model on {len(X_train):,} reviews..."
        )

        # A smaller feature set helps reduce training time.
        new_model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=12000,
                    ngram_range=(1, 1),
                    min_df=2,
                    stop_words="english",
                    dtype=float
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=300,
                    solver="liblinear",
                    random_state=RANDOM_SEED
                )
            )
        ])

        new_model.fit(X_train, y_train)

        status_label.config(
            text="Evaluating model and saving it..."
        )

        predictions = new_model.predict(X_test)
        new_accuracy = accuracy_score(y_test, predictions)

        # Save the trained model for future launches.
        joblib.dump(
            {
                "model": new_model,
                "accuracy": new_accuracy
            },
            MODEL_FILE
        )

        model = new_model
        accuracy = new_accuracy

        root.after(0, model_ready)

    except Exception as error:
        root.after(0, lambda err=str(error): show_error(err))


# -----------------------------------------
# MODEL READY
# -----------------------------------------

def model_ready():
    accuracy_label.config(
        text=f"Model Test Accuracy: {accuracy:.2%}"
    )

    status_label.config(
        text="Model ready! Enter a review below."
    )

    analyze_button.config(state="normal")


# -----------------------------------------
# ERROR HANDLING
# -----------------------------------------

def show_error(error):
    status_label.config(text="Could not prepare the model.")

    messagebox.showerror(
        "Model Error",
        "The model could not be loaded or trained.\n\n"
        + error
    )


# -----------------------------------------
# ANALYZE A REVIEW
# -----------------------------------------

def analyze_sentiment():
    if model is None:
        messagebox.showinfo(
            "Please Wait",
            "The model is still loading or training."
        )
        return

    review = review_box.get("1.0", tk.END).strip()

    if not review:
        messagebox.showwarning(
            "Empty Review",
            "Please enter a review first."
        )
        return

    prediction = model.predict([review])[0]

    probabilities = model.predict_proba([review])[0]
    classes = list(model.named_steps["classifier"].classes_)
    confidence = probabilities[classes.index(prediction)]

    if prediction == "positive":
        result_label.config(
            text="POSITIVE SENTIMENT",
            fg="#188038"
        )
    else:
        result_label.config(
            text="NEGATIVE SENTIMENT",
            fg="#D93025"
        )

    detail_label.config(
        text=f"Model confidence: {confidence:.1%}"
    )


# -----------------------------------------
# CLEAR THE INPUT
# -----------------------------------------

def clear_review():
    review_box.delete("1.0", tk.END)

    result_label.config(
        text="Sentiment will appear here",
        fg="#333333"
    )

    detail_label.config(text="")


# -----------------------------------------
# BUILD THE GUI
# -----------------------------------------

root = tk.Tk()
root.title("Sentiment Analysis Tool")
root.geometry("650x540")
root.resizable(False, False)
root.configure(bg="#F3F6FA")

heading = tk.Label(
    root,
    text="Sentiment Analysis Tool",
    font=("Arial", 23, "bold"),
    bg="#F3F6FA",
    fg="#183153"
)
heading.pack(pady=(25, 8))

subtitle = tk.Label(
    root,
    text="Analyze movie reviews and customer feedback",
    font=("Arial", 11),
    bg="#F3F6FA",
    fg="#555555"
)
subtitle.pack(pady=(0, 10))

accuracy_label = tk.Label(
    root,
    text="Model Test Accuracy: Loading...",
    font=("Arial", 12, "bold"),
    bg="#F3F6FA",
    fg="#2457C5"
)
accuracy_label.pack(pady=(0, 10))

status_label = tk.Label(
    root,
    text="Preparing model...",
    font=("Arial", 10),
    bg="#F3F6FA",
    fg="#555555"
)
status_label.pack(pady=(0, 10))

input_label = tk.Label(
    root,
    text="Enter your review:",
    font=("Arial", 12, "bold"),
    bg="#F3F6FA"
)
input_label.pack(anchor="w", padx=40)

review_box = tk.Text(
    root,
    height=7,
    width=65,
    font=("Arial", 11),
    wrap="word",
    bd=2,
    relief="groove"
)
review_box.pack(padx=35, pady=10)

button_frame = tk.Frame(root, bg="#F3F6FA")
button_frame.pack(pady=10)

analyze_button = tk.Button(
    button_frame,
    text="Analyze Sentiment",
    command=analyze_sentiment,
    font=("Arial", 11, "bold"),
    bg="#2457C5",
    fg="white",
    activebackground="#1745A5",
    activeforeground="white",
    width=18,
    state="disabled",
    cursor="hand2"
)
analyze_button.grid(row=0, column=0, padx=8)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_review,
    font=("Arial", 11),
    width=10,
    cursor="hand2"
)
clear_button.grid(row=0, column=1, padx=8)

result_label = tk.Label(
    root,
    text="Sentiment will appear here",
    font=("Arial", 18, "bold"),
    bg="#F3F6FA",
    fg="#333333"
)
result_label.pack(pady=(18, 5))

detail_label = tk.Label(
    root,
    text="",
    font=("Arial", 10),
    bg="#F3F6FA",
    fg="#555555"
)
detail_label.pack(pady=5)

footer = tk.Label(
    root,
    text="Model: TF-IDF + Logistic Regression",
    font=("Arial", 9),
    bg="#F3F6FA",
    fg="#777777"
)
footer.pack(side="bottom", pady=15)


# -----------------------------------------
# START TRAINING IN A BACKGROUND THREAD
# -----------------------------------------

threading.Thread(
    target=train_model,
    daemon=True
).start()

root.mainloop()

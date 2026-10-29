# Sentiment Analysis Tool

A beginner-friendly Python project that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative**. The application includes a graphical user interface (GUI) so users can enter a review and view the model's prediction.

## Features

- Predicts positive or negative sentiment from text.
- Provides a simple desktop GUI.
- Uses TF-IDF to convert text into numerical features.
- Uses Logistic Regression for classification.
- Trains and evaluates the model using IMDb movie reviews.
- Saves the trained model locally to reduce startup time on later runs.
- Displays model accuracy after evaluation.

## Technologies Used

- Python
- Tkinter for the graphical user interface
- scikit-learn for machine learning
- TF-IDF for text feature extraction
- Logistic Regression for sentiment classification
- Hugging Face Datasets for downloading the IMDb dataset
- Joblib for saving and loading the trained model

## Project Structure

```text
sentiment_analysis_tool/
├── sentiment_gui.py
├── download_dataset.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── .gitignore
└── screenshots/
```

The IMDb dataset and saved model are generated locally and are not included in this repository.

## Requirements

- Python 3.10 or a compatible version
- Internet connection for downloading the dataset and installing packages
- Windows, macOS, or Linux with a working Tkinter installation

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd sentiment_analysis_tool
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of your repository.

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
```

### 3. Install dependencies

On Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Download the IMDb dataset

```powershell
.\.venv\Scripts\python.exe download_dataset.py
```

This script downloads the dataset and prepares the review folders needed by the application. It requires an internet connection.

### 5. Run the application

```powershell
.\.venv\Scripts\python.exe sentiment_gui.py
```

The application trains the model on a selected sample during the first run and saves it locally. Later runs can load the saved model instead of training from scratch.

## How to Use

1. Start the application.
2. Wait for the model to finish loading or training.
3. Enter a movie review in the text box.
4. Click **Analyze Sentiment**.
5. Read the predicted sentiment and the displayed model accuracy.

## Screenshots

Screenshots of the running application will be added to the `screenshots/` folder.

## Dataset

The project uses the IMDb movie review dataset available through Hugging Face:

https://huggingface.co/datasets/stanfordnlp/imdb

The model is trained and evaluated on selected samples of the dataset, rather than necessarily using every available review. Results may vary with the data sample and software environment.

## Limitations

- The model predicts only positive or negative sentiment.
- Sarcasm, mixed opinions, and complex language may be classified incorrectly.
- Accuracy is measured on a selected test sample and does not guarantee correct predictions for every review.

## Future Enhancements

- Add a confidence score to predictions.
- Support additional languages.
- Add neutral sentiment classification.
- Build a web version using Streamlit or Flask.
- Analyze larger datasets and compare multiple machine learning models.

## Author

**Akash Jogi**

This project was developed for learning and demonstrating text classification and machine learning concepts.
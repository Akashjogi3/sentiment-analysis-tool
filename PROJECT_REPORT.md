# Project Report: Sentiment Analysis Tool

## 1. Abstract

The Sentiment Analysis Tool is a Python-based machine learning application that classifies movie reviews as positive or negative. It uses Natural Language Processing (NLP) techniques to process text and a Logistic Regression model to predict sentiment. The application provides a graphical user interface that allows users to enter a review and view the prediction. The IMDb movie review dataset is used to train and evaluate the model.

## 2. Introduction

Online reviews help people understand the opinions of other customers. Manually reading a large number of reviews can take considerable time. Sentiment analysis helps automate this task by identifying the opinion expressed in text.

This project demonstrates how machine learning can be applied to text classification. Users can enter a movie review through a desktop interface and receive a predicted sentiment.

## 3. Objectives

- Build a simple sentiment classification application.
- Learn basic Natural Language Processing concepts.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classifier.
- Evaluate the model using test data.
- Create a user-friendly desktop GUI.
- Save the trained model so it can be reused.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Tkinter | Desktop graphical user interface |
| scikit-learn | Text processing, model training and evaluation |
| TF-IDF | Converts review text into numerical features |
| Logistic Regression | Predicts positive or negative sentiment |
| Hugging Face Datasets | Downloads the IMDb dataset |
| Joblib | Saves and loads the trained model |

## 5. Dataset

The application uses the IMDb movie review dataset, which contains movie reviews labelled as positive or negative. The dataset is downloaded through the Hugging Face Datasets library.

The current application uses a selected sample of the training and test reviews to reduce training time. The exact sample size depends on the configuration in the source code.

## 6. System Design and Methodology

The application follows these steps:

1. Download and prepare the IMDb dataset.
2. Read positive and negative movie reviews.
3. Select training and testing samples.
4. Convert review text into TF-IDF features.
5. Train a Logistic Regression classifier.
6. Evaluate the classifier on test reviews.
7. Save the trained model locally.
8. Accept a review from the user through the GUI.
9. Display the predicted sentiment.

### Model workflow

```text
IMDb Dataset
     |
     v
Prepare Reviews
     |
     v
TF-IDF Feature Extraction
     |
     v
Logistic Regression
     |
     v
Model Evaluation
     |
     v
Save Trained Model
     |
     v
User Enters a Review
     |
     v
Predict Positive / Negative
```

## 7. Implementation

The application is implemented in Python. TF-IDF represents text using numerical features based on term importance. Logistic Regression learns patterns from labelled examples and predicts whether a new review is positive or negative.

Tkinter provides the desktop interface. Joblib stores the trained model so subsequent application launches can reuse it rather than retraining every time.

## 8. Testing and Results

The application should be tested with positive and negative movie reviews.

| Test case | Expected result |
|---|---|
| A clearly positive review | Positive |
| A clearly negative review | Negative |
| An empty input | Handled appropriately by the application |
| A second application launch after saving the model | Loads the saved model |
| A review with mixed opinions or sarcasm | May be classified incorrectly |

**Measured accuracy:** Add the actual test accuracy displayed by your application here: ______%.

Do not enter an estimated value. Record the value shown when the model is trained and evaluated.

## 9. Advantages

- Easy-to-use graphical interface.
- Demonstrates practical machine learning and NLP.
- Automates basic review classification.
- Saves the trained model to reduce subsequent startup time.
- Can be extended into a web application.

## 10. Limitations

- Classifies only positive and negative sentiment.
- May not understand sarcasm or mixed opinions.
- Results depend on the dataset, selected sample and model configuration.
- Requires an initial dataset download and model training.

## 11. Future Enhancements

- Add neutral sentiment classification.
- Display prediction confidence.
- Compare Logistic Regression with other classifiers.
- Support more languages.
- Build a web interface.
- Evaluate the model on additional review datasets.

## 12. Conclusion

The Sentiment Analysis Tool demonstrates how text processing and machine learning can be combined to classify movie reviews. It uses TF-IDF for feature extraction, Logistic Regression for classification, and Tkinter for the graphical interface. This project provides a foundation for further learning in NLP, machine learning and application development.

## 13. References

- IMDb dataset: https://huggingface.co/datasets/stanfordnlp/imdb
- scikit-learn documentation: https://scikit-learn.org/
- Python documentation: https://docs.python.org/3/
- Tkinter documentation: https://docs.python.org/3/library/tkinter.html
# Importing necessary libraries
import pandas as pd
from pyexpat import model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from fastapi import FastAPI, File, UploadFile,HTTPException
import pickle
from pred_variables import variables_dtypes
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
from sklearn.metrics import accuracy_score
from sklearn import metrics
from typing import List
import joblib

app = FastAPI()
class variables_dtypes(BaseModel):
    review: str


def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Endpoint to train SVM and Naive Bayes classifiers
@app.post("/binomail naive bayes model",)
async def upload_csv(file: UploadFile = File(...)):
    global svm_model, nb_bernoulli_model, nb_gaussian_model, multinomial

    # Read and process CSV file
    train_df = read_csv(file.file)
    print(train_df)
    x = train_df['review']
    y = train_df['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(x_train)

    model = BernoulliNB()
    model.fit(X_train_vectorized, y_train)
    vectorizer = TfidfVectorizer(max_features=5000)
    x_test = vectorizer.fit_transform(x_test)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {"model that trained is": "Binomial", "Accuracy of this model is ": accuracy}

@app.post("/svm model")
async def upload_csv(file: UploadFile = File(...)):
    global svm_model, nb_bernoulli_model, nb_gaussian_model, multinomial

    # Read and process CSV file
    train_df = read_csv(file.file)
    print(train_df)
    x = train_df['review']
    y = train_df['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(x_train)

    svm_classifier = SVC()
    svm_classifier.fit(X_train_vectorized, y_train)
    vectorizer = TfidfVectorizer(max_features=5000)
    x_test = vectorizer.fit_transform(x_test)
    y_pred = svm_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)


    return {"model that trained is": "Support vector machine", "Accuracy of this model is ": accuracy}


@app.post("/Gaussian Naive Bayes model")
async def upload_csv(file: UploadFile = File(...)):
    global svm_model, nb_bernoulli_model, nb_gaussian_model, multinomial

    # Read and process CSV file
    train_df = read_csv(file.file)
    print(train_df)
    x = train_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = train_df['Species']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # vectorizer = TfidfVectorizer(max_features=5000)
    # x_train= vectorizer.fit_transform(x_train)
    # x_train = x_train.toarray()

    gnb_classifier = GaussianNB()
    gnb_classifier.fit(x_train,y_train)
    # vectorizer = TfidfVectorizer(max_features=5000)
    # x_test = vectorizer.fit_transform(x_test)
    y_pred = gnb_classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)


    return {"model that trained is": "Gausian naive bayes", "Accuracy of this model is ": accuracy}


@app.post("/Multinomial Naive Bayes model")
async def upload_csv(file: UploadFile = File(...)):
    global svm_model, nb_bernoulli_model, nb_gaussian_model, multinomial

    # Read and process CSV file
    train_df = read_csv(file.file)
    print(train_df)
    x = train_df['review']
    y = train_df['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(x_train)

    mnb_classifier = MultinomialNB()
    mnb_classifier.fit(X_train_vectorized, y_train)
    vectorizer = TfidfVectorizer(max_features=5000)
    x_test = vectorizer.fit_transform(x_test)
    y_pred = mnb_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {"model that trained is": "multinomial naive bayes", "Accuracy of this model is ": accuracy}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
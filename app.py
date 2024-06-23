from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

app = Flask(__name__)
str1=""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def getvalue():
    name = request.form["name"]
    followers = request.form["followers"]
    following = request.form["following"]
    posts = request.form["posts"]
    private = request.form["private"]
    profile = request.form["profile"]
    link = request.form["link"]
    bio = request.form["bio"]
    len_name = len(request.form["name"])
    len_bio = len(request.form["bio"])

    if profile=="YES" or profile=="Yes" or profile=="yes":
        prof = 1
    else:
        prof = 0

    if private=="YES" or private=="Yes" or private=="yes":
        priv = 1
    else:
        priv = 0

    if link=="YES" or link=="Yes" or link=="yes":
        bio_link = 1
    else:
        bio_link = 0
    # len_name = len(request.form["name"])

    # loading the diabetes dataset to a pandas DataFrame
    diabetes_dataset = pd.read_csv(r"C:\Users\USER\Downloads\train.csv")




    # printing the first 5 rows of the dataset
    diabetes_dataset.head()


    diabetes_dataset.shape


    # getting the statistical measures of the data
    diabetes_dataset.describe()


    diabetes_dataset['fake'].value_counts()



    diabetes_dataset.groupby('fake').mean()


    # separating the data and labels
    X = diabetes_dataset.drop(columns = 'fake', axis=1)
    Y = diabetes_dataset['fake']



    scaler = StandardScaler()


    scaler.fit(X)


    standardized_data = scaler.transform(X)


    print(standardized_data)



    X = standardized_data
    Y = diabetes_dataset['fake']



    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)




    classifier = svm.SVC(kernel='linear')



    #training the support vector Machine Classifier
    classifier.fit(X_train, Y_train)


    # accuracy score on the training data
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)



    # accuracy score on the test data
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)



    input_data = (prof,0,0,0,0,len_bio,bio_link,priv,posts,followers,following)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    std_data = scaler.transform(input_data_reshaped)
    #print(std_data)

    prediction = classifier.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
        str1 = "The id is Real"
    elif (prediction[0] == 1):
        str1 = "The id is fake"

    return render_template("page.html", str2=str1)



if __name__ == "__main__":
    app.run(debug=True)
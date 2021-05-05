#This is Heroku Deployment Lectre

from flask import Flask, request, render_template
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import os
import joblib

print("Test")
print("Test 2")
print(os.getcwd())
path = os.getcwd()

with open('Models/knn_model.pkl', 'rb') as f:
    logistic = joblib.load(f)

with open('Models/dtc_model.pkl', 'rb') as f:
    randomforest = joblib.load(f)

with open('Models/svm_rbf_model.pkl', 'rb') as f:
    svm_model = joblib.load(f)


def get_predictions(age, sex, cp, trestbps, chol, fbs, restecg, thalach,exang, oldpeak, slope, ca, thal, req_model):
    mylist = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,exang, oldpeak, slope, ca, thal]
    mylist = [float(i) for i in mylist]
    vals = [mylist]

    if req_model == 'KNN':
        #print(req_model)
        return logistic.predict(vals)[0]

    elif req_model == 'DecisionTree':
        #print(req_model)
        return randomforest.predict(vals)[0]

    elif req_model == 'SVM':
        #print(req_model)
        return svm_model.predict(vals)[0]
    else:
        return "Cannot Predict"


app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        cp= request.form['cp']
        trestbps= request.form['trestbps']
        chol= request.form['chol']
        fbs= request.form['fbs']
        restecg= request.form['restecg']
        thalach= request.form['thalach']
        exang= request.form['exang']
        oldpeak= request.form['oldpeak']
        slope= request.form['slope']
        ca= request.form['ca']
        thal= request.form['thal']
        req_model = request.form['req_model']

        target = get_predictions(age, sex, cp, trestbps, chol, fbs, restecg, thalach,exang, oldpeak, slope, ca, thal, req_model)
        risk_level=0
        if target==0:
            risk_level = 'No heart disease'
        if target == 1:
            risk_level = 'There is a high risk of having a heart disease'
        return render_template('home.html', target = target, risk_level = risk_level)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
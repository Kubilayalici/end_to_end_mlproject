from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application=Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            school=request.form['school'],
            sex=request.form['sex'],
            age=int(request.form['age']),
            address=request.form['address'],
            famsize=request.form['famsize'],
            Pstatus=request.form['Pstatus'],
            Medu=int(request.form['Medu']),
            Fedu=int(request.form['Fedu']),
            Mjob=request.form['Mjob'],
            Fjob=request.form['Fjob'],
            reason=request.form['reason'],
            guardian=request.form['guardian'],
            traveltime=int(request.form['traveltime']),
            studytime=int(request.form['studytime']),
            failures=int(request.form['failures']),
            schoolsup=request.form['schoolsup'],
            famsup=request.form['famsup'],
            paid=request.form['paid'],
            activities=request.form['activities'],
            nursery=request.form['nursery'],
            higher=request.form['higher'],
            internet=request.form['internet'],
            romantic=request.form['romantic'],
            famrel=int(request.form['famrel']),
            freetime=int(request.form['freetime']),
            goout=int(request.form['goout']),
            Dalc=int(request.form['Dalc']),
            Walc=int(request.form['Walc']),
            health=int(request.form['health']),
            absences=int(request.form['absences']),
            G1=int(request.form['G1']),
            G2=int(request.form['G2']))

                
        pred_df = data.to_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results= results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0")
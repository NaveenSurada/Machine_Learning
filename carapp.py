import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
import sklearn

app=Flask(__name__)

@app.route('/')
def Home():
    return render_template('indexcar.html')

@app.route('/predict',methods=['POST','GET'])
def results():
    wheelbase= float(request.form['wheelbase'])
    carlength= float(request.form['carlength'])
    carwidth= float(request.form['carwidth'])
    carheight= float(request.form['carheight'])
    curbweight= float(request.form['curbweight'])
    enginesize= float(request.form['enginesize'])
    stroke= float(request.form['stroke'])
    compressionratio= float(request.form['compressionratio'])
    horsepower= float(request.form['horsepower'])
    peakrpm= float(request.form['peakrpm'])

    X= np.array([[wheelbase, carlength, carwidth, carheight, curbweight, enginesize, stroke, compressionratio, horsepower, peakrpm]])
    model= pickle.load(open('modelcar.pkl','rb'))
    Y_prediction = model.predict(X)
    return jsonify({'Model Prediction':float(Y_prediction)})

if __name__=='__main__':
    app.run(debug= True, port=1010)


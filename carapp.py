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
    x1= float(request.form['wheelbase'])
    x2= float(request.form['carlength'])
    x3= float(request.form['carwidth'])
    x4= float(request.form['carheight'])
    x5= float(request.form['curbweight'])
    x6= float(request.form['enginesize'])
    x7= float(request.form['stroke'])
    x8= float(request.form['compressionratio'])
    x9= float(request.form['horsepower'])
    x10= float(request.form['peakrpm'])

    X= np.array([[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]])
    model= pickle.load(open('modelcar.pkl','rb'))
    Y_prediction = model.predict(X)
    return jsonify({'Model Prediction':float(Y_prediction)})

if __name__=='__main__':
    app.run(debug= True, port=1010)


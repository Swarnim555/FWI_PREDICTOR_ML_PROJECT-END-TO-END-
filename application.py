import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# WE MADE A WEB APPLICATION

application=Flask(__name__)
app=application

# IMPORT RIDGE AND SCALER PICKLE

with open('models/scaler.pkl','rb') as f:
   scaler=pickle.load(f)
with open('models/ridge_reg.pkl','rb') as f:
    ridge_reg=pickle.load(f)

#HOMEPAGE
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home',methods=["GET","POST"])
def home():
    if request.method=="POST":
        try:

            Temperature=float(request.form.get('Temperature'))
            RH=float(request.form.get('RH'))
            Ws=float(request.form.get('Ws'))
            Rain=float(request.form.get('Rain'))
            FFMC=float(request.form.get('FFMC'))
            DMC=float(request.form.get('DMC'))
            ISI=float(request.form.get('ISI'))
            Classes=float(request.form.get('Classes'))
            Region=float(request.form.get('Region'))
            DC=float(request.form.get('DC'))
            Month=float(request.form.get('Month'))
            Day=float(request.form.get('Day'))
        
        except(TypeError,ValueError) as e:
            return render_template('home.html',error='Please fill in all fields with valid numbers.')
    
        new_data=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,Classes,Region,Month,Day]])

        result=ridge_reg.predict(new_data)
        print(result)

        return render_template('home.html',results=result[0])
    
    else:
        return render_template('home.html',results=None)

if __name__=="__main__":
    app.run(debug=True)

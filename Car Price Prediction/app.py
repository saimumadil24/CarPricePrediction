from flask import Flask, render_template, request
import pickle as pk
import pandas as pd
import CarPrice
app=Flask(__name__)

@app.route('/',methods=['POST','GET'])
def main():
    if request.method=='POST':
        model=pk.load(open(r'C:\Users\Saimum Adil Khan\OneDrive\Desktop\Flask\Car Price Prediction\CarPrice.pkl','rb'))
        Mileage=float(request.form.get('mileage'))
        Age=float(request.form.get('age'))
        car_make=request.form.get('car_make')
        Audi=1 if car_make=='audi' else 0
        BMW=1 if car_make=='bmw' else 0
        Mercedez=1 if car_make=='mercedez' else 0
        Toyota=1 if car_make=='toyota' else 0
        pred_data=pd.DataFrame(columns=CarPrice.X_train.columns)
        pred_data.loc[0]=[Mileage,Age,BMW,Mercedez,Toyota]
        prediction=model.predict(pred_data).round(2)
        prediction=f"{prediction[0]} USD"
    else:
        prediction=''
    return render_template('home.html',output=prediction)

if __name__=='__main__':
    app.run(debug=True)
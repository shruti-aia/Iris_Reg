from flask import Flask , render_template , jsonify ,request
from project_app.utils import IrisPrediction
import pickle
import json
import config

app=Flask(__name__)

@app.route("/")

def base():
    return render_template("home.html")

@app.route("/predict",methods=["POST"])

def home():
    SepalLengthCm = request.form["SepalLengthCm"]
    SepalWidthCm = request.form["SepalWidthCm"]
    PetalLengthCm = request.form["PetalLengthCm"]
    PetalWidthCm =request.form["PetalWidthCm"]

    get_len=IrisPrediction(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)

    result=get_len.predict_len()

    #return jsonify ({"result":f"the length of flower is {result}"})

    return render_template("after.html",data=result)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=config.PORT_NO,debug=True)
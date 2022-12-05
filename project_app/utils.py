import pandas as pd
import numpy as np
import config
import pickle
import json


class IrisPrediction():
    def __init__(self,SepalLengthCm,SepalWidthCm ,PetalLengthCm ,PetalWidthCm ):
        self.SepalLengthCm=SepalLengthCm
        self.sepalWidthCm = SepalWidthCm
        self.PetalLengthCm= PetalLengthCm
        self.PetalWidthCm=PetalWidthCm
        

    def load_model(self):
        with open(config.model_file_path,"rb") as f:
            self.dt_model=pickle.load(f)

        with open (config.project_data_path,"r") as f:
            self.project_json=json.load(f)


    def predict_len(self):
        self.load_model()
        test_array=np.zeros(len(self.project_json["columns"]))
        test_array[0]=self.SepalLengthCm
        test_array[1]=self.sepalWidthCm
        test_array[2]=self.PetalLengthCm
        test_array[3]=self.PetalWidthCm
       

        prediction=self.dt_model.predict([test_array])

        return prediction





        
# -- coding: utf-8
import pickle as pk
import warnings
import joblib
import pandas as pd


class BlurryModel:
    def __init__(self, scaler_path, pca_path, model_path):
        self.loaded_scaler = pk.load(open(scaler_path, 'rb'))
        self.loaded_pca = pk.load(open(pca_path, 'rb'))
        self.loaded_svm = joblib.load(model_path)

    def predict(self, pandas_data):
        df = pd.DataFrame(pandas_data)
        # print(df.shape)
        warnings.simplefilter('ignore')
        df_scaled = self.loaded_scaler.transform(df)
        df = pd.DataFrame(df_scaled)
        # print(df.shape)

        df_pca = self.loaded_pca.transform(df)
        df = pd.DataFrame(df_pca)
        # print(df.shape)

        prediction = self.loaded_svm.predict(df)
        # print(prediction)
        return prediction[0]

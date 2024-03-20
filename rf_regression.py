import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

if __name__=="__main__":

    dataset = (r'C:\Users\Niti\NEXT HIKES\PROJECT 6\final_df_for_dashboard.csv')

    with mlflow.start_run():
        mlflow.log_param("dataset", dataset)

        df = pd.read_csv(dataset)

        X = df.drop('Sales', axis=1)
        y = df['Sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)

        # score model
        mean_accuracy = model.score(X_train, y_train)
        print(f"Mean Accuracy: {mean_accuracy}")

        mlflow.log_metric("mean accuracy", mean_accuracy)

        # export model
        mlflow.sklearn.log_model(model, "model")
        run_id = mlflow.active_run().info.run_uuid
        print(f"Model saved in run {run_id}") 


#input_data = [[1,1,1,555,1,31,31,7,2015]]

# Function to make the prediction
#def predict(input_data):
  # apply the neccesary preprocessing
    
# make the prediction
    #prediction = model.predict(input_data)
    #return prediction

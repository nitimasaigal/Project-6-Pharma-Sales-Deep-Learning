import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\Niti\NEXT HIKES\PROJECT 6\final_df_for_dashboard.csv', index_col=None)
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

X = df.drop('Sales', axis=1)
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model,'model.pkl')

#Load the model
model = joblib.load('model.pkl')

input_data = [[555,1,8,1,4,2015,1,31,1]]

# Function to make the prediction
def predict(input_data):
  # apply the neccesary preprocessing
    
    input_df = pd.DataFrame([input_data])
    
# make the prediction
    prediction = model.predict(input_df)
    return prediction

prediction = predict(input_data)
print("Predicted Sales:", prediction)
    


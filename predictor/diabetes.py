import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import os
from django.conf import settings

csv_path = os.path.join(settings.BASE_DIR, 'predictor', 'diabetes.csv')
df = pd.read_csv(csv_path)

X = df[
['Pregnancies', 'Glucose', 'BloodPressure', 
'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
].to_numpy()
Y = df['Outcome'].to_numpy()
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)


model = svm.SVC(C=0.1, kernel='linear')
model.fit(X_train, Y_train)


def predict_diabetic(parametres):
    scaled_params = scaler.transform(parametres)
    prediction = model.predict(scaled_params)
    return prediction


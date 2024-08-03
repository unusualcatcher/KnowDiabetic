from django.shortcuts import render
from .diabetes import predict_diabetic
import numpy as np

def home(request):
    if request.method == "POST":
        pregnancies = float(request.POST.get('pregnancies'))
        glucose = float(request.POST.get('glucose'))
        blood_pressure = float(request.POST.get('bloodPressure'))
        skin_thickness = float(request.POST.get('skinThickness'))
        insulin = float(request.POST.get('insulin'))
        bmi = float(request.POST.get('bmi'))
        diabetes_pedigree_function = float(request.POST.get('diabetesPedigreeFunction'))
        age = float(request.POST.get('age'))

        arr = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])
        print("Array:", arr)
        prediciton = predict_diabetic(np.array([arr]))
        return render(request, 'predictor/index.html', context={'output':prediciton[0]})
    return render(request, 'predictor/index.html', context={})

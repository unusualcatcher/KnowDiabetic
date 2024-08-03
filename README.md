# Diabetes Prediction Web App

This is a Django web application designed to predict whether a user has diabetes based on input data using a Support Vector Classification (SVC) model.

## Features

- User-friendly web interface for inputting data
- Predictive model based on SVC
- Real-time prediction results

## Technologies Used

- Django
- Pandas
- NumPy
- Scikit-learn

## Requirements

To run this project, you need to install the following Python packages:

- `pandas`
- `django`
- `numpy`
- `scikit-learn`

## Downloading and Running the App from a ZIP File
1. Download the ZIP file from the GitHub repository.

2. Extract the ZIP file to your desired location.

3. Navigate to the extracted directory:
`cd path/to/extracted-directory`

4. Create and activate a virtual environment:
`python -m venv venv`
`source venv/bin/activate`

 On Windows use `venv\Scripts\activate`

6. Install the required packages:
`python -m pip install scikit-learn django pandas numpy`

7. Start the Django development server:
`python manage.py runserver`

8. Open your browser and go to http://127.0.0.1:8000/ to access the application.

## Acknowledgements
The Machine Learning model in this project was trained using the data set provided by kaggle. It is avaible at: https://www.kaggle.com/datasets/mathchi/diabetes-data-set


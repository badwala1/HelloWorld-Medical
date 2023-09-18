import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def warn(*args, **kwargs):
    pass


warnings.warn = warn


obesity_pd = pd.read_csv('Data-sets/obesity.csv')
obesity_pd.drop('ID', axis=1, inplace=True)
obesity_y = obesity_pd['Label']
obesity_pd.drop('Label', axis=1, inplace=True)
obesity_x = obesity_pd
obesity_rf = RandomForestClassifier(n_estimators=15)

obesity_x_train, obesity_x_test, obesity_y_train, obesity_y_test = train_test_split(
    obesity_x, obesity_y, test_size=0.2)
obesity_rf.fit(obesity_x_train, obesity_y_train)

heart_pd = pd.read_csv('Data-sets\heart.csv')
heart_pd.drop('bad', axis=1, inplace=True)
heart_y = heart_pd['DEATH_EVENT']
heart_pd.drop('DEATH_EVENT', axis=1, inplace=True)
heart_x = heart_pd
heart_rf = RandomForestClassifier(n_estimators=15)

heart_x_train, heart_x_test, heart_y_train, heart_y_test = train_test_split(
    heart_x, heart_y, test_size=0.2)
heart_rf.fit(heart_x_train, heart_y_train)

diabetes_pd = pd.read_csv('Data-sets\diabetes.csv')
diabetes_y = diabetes_pd['diabetes']
diabetes_pd.drop('diabetes', axis=1, inplace=True)
diabetes_x = diabetes_pd
diabetes_rf = RandomForestClassifier(n_estimators=15)

diabetes_x_train, diabetes_x_test, diabetes_y_train, diabetes_y_test = train_test_split(
    diabetes_x, diabetes_y, test_size=0.2)
diabetes_rf.fit(diabetes_x_train, diabetes_y_train)

kidney_pd = pd.read_csv('Data-sets/kidney.csv')
kidney_y = kidney_pd['Class']
kidney_pd.drop('Class', axis=1, inplace=True)
kidney_x = kidney_pd
kidney_rf = RandomForestClassifier(n_estimators=15)

kidney_x_train, kidney_x_test, kidney_y_train, kidney_y_test = train_test_split(
    kidney_x, kidney_y, test_size=0.2)
kidney_rf.fit(kidney_x_train, kidney_y_train)

f = open("logo.txt", "r")
print(f.read())

print("Welcome to BoilerMD, use this guide if you need help for data input: https://docs.google.com/document/d/1lKU6VU5_MCDUozAH2NHf50LTmbPE0ScfqfzKmNnWpIw")
print("Please choose the disease you want to predict from these: ")
print("1. Obesity")
print("2. Heart Failure")
print("3. Diabetes")
print("4. Kidney Disease")

choice = int(input("Enter your choice: "))
match choice:
    case 1:
        print("Please enter the following data for obesity classification: ")
        print("Age, Gender, Height, Weight, BMI")
        print("If you do not have an answer to one of the data points, use these values: 39,0,175,80,29")
        input = input("Enter your data: ")
        output = obesity_rf.predict(np.array(input.split(',')).reshape(1, -1))
        if output[0] == 0:
            print("Underweight")
        elif output[0] == 1:
            print("Normal Weight")
        elif output[0] == 2:
            print("Overweight")
        elif output[0] == 3:
            print("Obese")
        else:
            print("Problem occured")
        chances = obesity_rf.predict_proba(
            np.array(input.split(',')).reshape(1, -1))
        print("Chances of being underweight: ", round(chances[0][0], 3))
        print("Chances of being normal weight: ", round(chances[0][1], 3))
        print("Chances of being overweight: ", round(chances[0][2], 3))
        print("Chances of being obese: ", round(chances[0][3], 3))
    case 2:
        print("Please enter the following data for heart failure classification: ")
        print('age, anaemia, creatinine phosphokinase, diabetes, ejection fraction, high blood pressure, platelets,serum creatinine, serum sodium, gender, smoking, time')
        print("If you do not have an answer to one of the data points, use these values: 42,0,130,0,60,0,275000,0.9,140,0,0,130")
        input = input("Enter your data: ")
        output = heart_rf.predict(np.array(input.split(',')).reshape(1, -1))
        if output[0] == 0:
            print("Safe from heart failure")
        elif output[0] == 1:
            print("At risk of heart failure")
        else:
            print("Problem occured")
        chances = heart_rf.predict_proba(
            np.array(input.split(',')).reshape(1, -1))
        print("Chances of being safe from heart failure: ",
              round(chances[0][0], 3))
        print("Chances of being at risk of heart failure: ",
              round(chances[0][1], 3))
    case 3:
        print("Please enter the following data for diabetes classification: ")
        print('gender, age, hypertension, heart disease, smoking history, bmi, HbA1c level, blood glucose level')
        print("If you do not have an answer to one of the data points, use these values: 0,39,0,0,0,29,5,95")
        input = input("Enter your data: ")
        output = diabetes_rf.predict(np.array(input.split(',')).reshape(1, -1))
        if output[0] == 0:
            print("Safe from diabetes")
        elif output[0] == 1:
            print("At risk of diabetes")
        else:
            print("Problem occured")
        chances = diabetes_rf.predict_proba(
            np.array(input.split(',')).reshape(1, -1))
        print("Chances of being safe from diabetes: ", round(chances[0][0], 3))
        print("Chances of being at risk of diabetes: ",
              round(chances[0][1], 3))
    case 4:
        print("Please enter the following data for kidney disease classification: ")
        print("Bp, Sg, Al, Su, Rbc, Bu, Sc, Sod, Pot, Hemo, Wbcc, Rbcc, Htn")
        print("If you do not have an answer to one of the data points, use these values: 76,1,1,0,1,30,1,137,4.4,15,7000,5,0")
        input = input("Enter your data: ")
        output = kidney_rf.predict(np.array(input.split(',')).reshape(1, -1))
        if output[0] == 0:
            print("Safe from kidney disease")
        elif output[0] == 1:
            print("At risk of kidney disease")
        else:
            print("Problem occured")
        chances = kidney_rf.predict_proba(
            np.array(input.split(',')).reshape(1, -1))
        print("Chances of being safe from kidney disease: ",
              round(chances[0][0], 3))
        print("Chances of being at risk of kidney disease: ",
              round(chances[0][1], 3))
    case _:
        print("Invalid choice")

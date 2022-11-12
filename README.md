# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Gayathri A
RegisterNumber:  212221230028
*/

import pandas as pd
import numpy as np
import matplotlib as plt
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:

![41](https://user-images.githubusercontent.com/94154854/201481526-5021455a-230f-41b2-986f-78c9d4be1aba.png)

![42](https://user-images.githubusercontent.com/94154854/201481532-43157ac2-e9b7-4814-8503-4e2e2ffb2c77.png)

![43](https://user-images.githubusercontent.com/94154854/201481547-850a368d-118f-4ee5-b412-4646c86b219e.png)

![44](https://user-images.githubusercontent.com/94154854/201481557-583d2c1f-cd9b-48f1-9812-b669978e56dd.png)

![45](https://user-images.githubusercontent.com/94154854/201481570-86672590-b2f3-494d-8888-dbb2474cf84b.png)

![46](https://user-images.githubusercontent.com/94154854/201481578-7d27cf3c-5ce1-445f-8402-09a34cbbdd4e.png)

![47](https://user-images.githubusercontent.com/94154854/201481589-962cc061-c48b-4fe3-afe4-d6a6af75cf21.png)

![48](https://user-images.githubusercontent.com/94154854/201481594-a0b69aa3-7264-4f17-be92-84058edb3445.png)

![49](https://user-images.githubusercontent.com/94154854/201481599-21429887-a2a2-4ab2-8af5-85d0a70764c9.png)

![50](https://user-images.githubusercontent.com/94154854/201481604-2dfd3daf-f4f4-4927-b9d9-ea8946e748e6.png)

![511](https://user-images.githubusercontent.com/94154854/201481615-9ad54779-46a8-41a5-89bd-25234bc808ac.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

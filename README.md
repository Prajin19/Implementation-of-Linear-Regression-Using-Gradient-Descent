# Implementation of Linear Regression Using Gradient Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the dataset, extract features and target values, convert them to float, and apply standard scaling.
2. Define a linear regression function that initializes parameters and iteratively updates them using gradient descent.
3. Scale new input data, append a bias term, and use the trained model to make a prediction.
4. Inversely transform the scaled prediction to return it to the original scale and print the result.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Prajin S
RegisterNumber: 212223230151
*/
```
```Python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta_=learning_rate*(1/len(X1))*X.T.dot(errors)
        pass
    return theta
data=pd.read_csv('50_Startups.csv')
data.head()
X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
X1_Scaled
Y1_Scaled
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![image](https://github.com/user-attachments/assets/19143df6-cfef-4355-91b6-7eaa1327ca5a)

![image](https://github.com/user-attachments/assets/beaec50f-2527-48f7-8f0e-656866b6e94e)

![image](https://github.com/user-attachments/assets/e6582381-e2af-40ac-b3bd-03626cba6983)

![image](https://github.com/user-attachments/assets/976035ae-59a4-45e8-b749-2dd6e1eb4d40)

![image](https://github.com/user-attachments/assets/9db0e259-e32d-4428-9058-8080db393abc)

![image](https://github.com/user-attachments/assets/41a92dea-54cb-4412-99db-4cca21696d07)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

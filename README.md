# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Load the California housing dataset, extract features (first three columns) and targets (target variable and sixth column), and split the data into training and testing sets. 2.Data Scaling: Standardize the feature and target data using StandardScaler to enhance model performance. 3.Model Training: Create a multi-output regression model with SGDRegressor and fit it to the training data. 4.Prediction and Evaluation: Predict values for the test set using the trained model, calculate the mean squared error, and print the predictions along with the squared error.
## Program:
```

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: KARTHIKEYAN A
RegisterNumber:  212225230131

```
```
#SGD-ex4
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("house.csv")
#print(data.columns)
data.columns = data.columns.str.strip()
# Features (inputs)
X = data[['Size', 'Bedrooms']]

# Targets (outputs)
y_price = data['Price']
y_occ = data['Occupants']

# Scaling (important for SGD)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Models
price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)

# Train models
price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)

# Input
size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))

# Scale input
new_data = scaler.transform([[size, bed]])

# Prediction
pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
```
## Output:
```
Enter the size of the room: 1300
Enter the number of bedrooms: 3
Price: 375130.7050704837
Occupants: 4
```

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

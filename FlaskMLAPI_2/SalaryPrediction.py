import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("./SalaryData.csv")

#Check the loaded dataset
df.head()
df.shape
df.isnull().values.any()

#Split data into train/test
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
df_copy = train_set.copy()

test_set_full = test_set.copy()
test_set = test_set.drop(["Salary"], axis=1)

test_set.head()
train_labels = train_set["Salary"]

train_labels.head()

train_set_full = train_set.copy()
train_set = train_set.drop(["Salary"], axis=1)

train_set.head()

#Get the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(train_set, train_labels)

salary_pred = lin_reg.predict(test_set)
salary_pred

#Analyze results
print("Coefficients: ", lin_reg.coef_)
print("Intercept: ", lin_reg.intercept_)

lin_reg.score(test_set, test_set_full["Salary"])

#Model persistance with joblib
from sklearn.externals import joblib
joblib.dump(lin_reg, "linear_regression_model.pkl")


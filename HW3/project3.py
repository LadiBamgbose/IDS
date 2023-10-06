import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate
LifeExpectancy = pd.read_csv("~/Downloads/Life_Expectancy_Mac.csv")
LifeExpectancy = LifeExpectancy.query("Population != 0")

# Calculate the mean of the columns
meanL = LifeExpectancy['Life_Expectancy '].mean()
meanA = LifeExpectancy['Adult_Mortality'].mean()
meanAl = LifeExpectancy['Alcohol'].mean()
meanP = LifeExpectancy['Percentage_Expenditure'].mean()
meanB = LifeExpectancy[' BMI '].mean()
meanT = LifeExpectancy['Total_Expenditure'].mean()
meanG = LifeExpectancy['GDP'].mean()
meanPo = LifeExpectancy['Population'].mean()
meanS = LifeExpectancy['Schooling'].mean()



# Replace null values in "Life_Expectancy" with the mean
LifeExpectancy['Life_Expectancy '].fillna(meanL, inplace=True)
LifeExpectancy['Adult_Mortality'].fillna(meanA, inplace=True)
LifeExpectancy['Alcohol'].fillna(meanAl, inplace=True)
LifeExpectancy['Percentage_Expenditure'].fillna(meanP, inplace=True)
LifeExpectancy[' BMI '].fillna(meanB, inplace=True)
LifeExpectancy['Total_Expenditure'].fillna(meanT, inplace=True)
LifeExpectancy['GDP'].fillna(meanG, inplace=True)
LifeExpectancy['Population'].fillna(meanPo, inplace=True)
LifeExpectancy['Schooling'].fillna(meanS, inplace=True)


# Verify the changes
# print("Count of null values in LE:", LifeExpectancy['Life_Expectancy '].isnull().sum())
# print("Count of null values in A_M:", LifeExpectancy['Adult_Mortality'].isnull().sum())
# print("Count of null values in Alcohol:", LifeExpectancy['Alcohol'].isnull().sum())
# print("Count of null values in P_E:", LifeExpectancy['Percentage_Expenditure'].isnull().sum())
# print("Count of null values in BMI:", LifeExpectancy[' BMI '].isnull().sum())
# print("Count of null values in T_E:", LifeExpectancy['Total_Expenditure'].isnull().sum())
# print("Count of null values in GDP:", LifeExpectancy['GDP'].isnull().sum())
# print("Count of null values in Population:", LifeExpectancy['Population'].isnull().sum())
# print("Count of null values in Schooling:", LifeExpectancy['Schooling'].isnull().sum())

#plotting Adult Mortality rates vs Life Expectancy
x = LifeExpectancy['Schooling'].values.reshape(-1,1)
y = LifeExpectancy['Life_Expectancy '].values


model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)

# Plot the data points and the linear regression line
plt.scatter(x, y, label='Data')
plt.plot(x, y_pred, color='red', label='Linear Regression Line')
plt.xlabel('Schooling')
plt.ylabel('Life Expectancy')
plt.title('Linear Regression Model')
plt.show()

print("The slope: ", model.coef_)
print("The intercept: ", model.intercept_)

print("MSE: ", mean_squared_error(y, y_pred))
print("R2: ", r2_score(y, y_pred))














import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
df = pd.read_csv('recovery_score_dataset.csv')

x=df.drop(['recovery_score'],axis=1)
y=df['recovery_score']

#Linear regression model.
x_train,x_temp,y_train,y_temp=train_test_split(x,y,test_size=0.2,random_state=42)

x_valid,x_test,y_valid,y_test=train_test_split(x_temp,y_temp,test_size=0.5,random_state=42)

# clf=LinearRegression().fit(x_train,y_train)

# print(round(clf.score(x_train,y_train),2))

# new_data = pd.DataFrame({
#     "sleep_hours": [7.6],
#     "resting_heart_rate": [51],
#     "previous_workout_intensity": [6],
#     "muscle_soreness": [2],
#     "water_intake_liters": [2.5]
# })
# print(np.round(clf.predict(new_data),2))

# print(clf.predict([[7.6,51,6,2,2.5]]))

# print(np.round(clf.score(x_valid,y_valid),2))

# print(np.round(clf.score(x_test,y_test),2))

# pred = clf.predict(x_test)

# print("MAE :", np.round(mean_absolute_error(y_test, pred), 2))
# print("MSE :", np.round(mean_squared_error(y_test, pred), 2))
# print("RMSE:", np.round(np.sqrt(mean_squared_error(y_test, pred)), 2))
# print("R²  :", np.round(r2_score(y_test, pred), 2))

# print(clf.coef_)
# print(clf.intercept_)

# print(df.corr()["recovery_score"].sort_values(ascending=False))


# print(np.round(clf.score(x_test,y_test),2))

#Random Forest Regressor model.
rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(x_train, y_train)

rfpred=rf.predict(x_test)
# print(np.round(rf.predict(new_data), 2))

print("RF MAE :", np.round(mean_absolute_error(y_test, rfpred), 2))
print("RF MSE :", np.round(mean_squared_error(y_test, rfpred), 2))
print("RF RMSE:", np.round(np.sqrt(mean_squared_error(y_test, rfpred)), 2))
print("RF R²  :", np.round(r2_score(y_test, rfpred), 2))

print(np.round(np.average(rfpred),2))


#Random forest refressor has slight better score in mae,mse,rmse difference with the linear regression model.
#So based on the evaluation metrics, I will use the Random Forest Regressor model for the recovery prediction model.

joblib.dump(rf, 'recovery_model.pkl')
print("Random Forest Regressor model saved as 'recovery_model.pkl'")


#Tested the Linea regresssion without some features and the model performance was not good 
# so I kept all the features in the model.


# x_new=x.drop('muscle_soreness',axis=1)
# x_train, x_temp, y_train, y_temp = train_test_split(
#     x_new, y, test_size=0.2, random_state=42
# )

# x_valid, x_test, y_valid, y_test = train_test_split(
#     x_temp, y_temp, test_size=0.5, random_state=42
# )

# clf = LinearRegression()
# clf.fit(x_train, y_train)

# print(np.round(clf.predict(new_data), 2))


# pred = clf.predict(x_test)

# print("MAE :", np.round(mean_absolute_error(y_test, pred), 2))
# print("MSE :", np.round(mean_squared_error(y_test, pred), 2))
# print("RMSE:", np.round(np.sqrt(mean_squared_error(y_test, pred)), 2))
# print("R²  :", np.round(r2_score(y_test, pred), 2))
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import joblib

# Load data from CSV file
data = pd.read_csv('IR_latency_lut.csv')

# Split data into features (X) and target labels (y)
if 1:
    ct = ColumnTransformer([('encoder', OneHotEncoder(), [2,3])], remainder='passthrough')
    X = ct.fit_transform(data.iloc[:, :-1].values)
else:
    X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create SVM regression model and fit to training data
svm = SVR(kernel='linear', C=1, epsilon=0.1)
svm.fit(X_train, y_train)

# Predict on test data
y_pred = svm.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

print('Mean Squared Error:', mse)

# Save trained model to file
joblib.dump(svm, 'ir_svm_model.pkl')

# Load saved model from file
# loaded_model = joblib.load('svm_model.pkl')

"""
CM -- Mean Squared Error: 0.002586174745984428
RS -- Mean Squared Error: 0.005343032788250735
"""

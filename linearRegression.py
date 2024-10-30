import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Nairobi)Office_Price_Ex(1).csv')
X = data['SIZE'].values
y = data['PRICE'].values
#
# Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function
def gradient_descent(X, y, m, c, learning_rate):
    N = len(y)
    y_pred = m * X + c
    # Calculate gradients
    dm = (-2 / N) * sum(X * (y - y_pred))
    dc = (-2 / N) * sum(y - y_pred)
    # Update weights
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

# Initialize parameters
m = np.random.rand()
c = np.random.rand()
learning_rate = 0.0001
epochs = 10

# Training process
for epoch in range(epochs):
    y_pred = m * X + c
    mse = mean_squared_error(y, y_pred)
    print(f'Epoch {epoch + 1}, MSE: {mse:.4f}')
    m, c = gradient_descent(X, y, m, c, learning_rate)

# Plotting the final line of best fit
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, m * X + c, color='red', label='Best Fit Line')
plt.xlabel('Office Size')
plt.ylabel('Office Price')
plt.legend()
plt.show()

# Predicting for office size = 100 sq.ft
office_size = 100
predicted_price = m * office_size + c
print(f'Predicted office price for 100 sq.ft: {predicted_price:.2f}')




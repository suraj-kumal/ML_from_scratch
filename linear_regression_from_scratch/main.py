import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('studytime_score_dataset.csv')

def visualize():
    plt.scatter(data['studytime'], data['score'])
    plt.xlabel('Study Time')
    plt.ylabel('Score')
    plt.title('Study Time vs Score')
    plt.show()

# Mean squared error
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i]['studytime']
        y = points.iloc[i]['score']
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i]['studytime']
        y = points.iloc[i]['score']
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - (L * m_gradient)
    b = b_now - (L * b_gradient)

    return m, b

m = 0
b = 0
L = 0.0001
epochs = 1000


for i in range(epochs):
    if i % 20 == 0:
        print(f"Epoch {i}: Loss = {loss_function(m, b, data)}")
    m, b = gradient_descent(m, b, data, L)

print(f"Final parameters: m = {m}, b = {b}")


def output_graph():
    plt.scatter(data['studytime'], data['score'])
    x_values = np.linspace(data['studytime'].min(), data['studytime'].max(), 100)
    y_values = m * x_values + b
    plt.plot(x_values, y_values, color='red')
    plt.xlabel('Study Time')
    plt.ylabel('Score')
    plt.title('Study Time vs Score with Regression Line')
    plt.show()


visualize()
output_graph()

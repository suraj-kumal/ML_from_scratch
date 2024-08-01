import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

data = pd.read_csv("dataset.csv")

def visualize():
    has_car = data[data["has_car"] == 1]
    no_car = data[data["has_car"] == 0]
    plt.scatter(has_car["monthly_salary"],has_car["num_children"],color="blue",label="Has Car")
    plt.scatter(no_car["monthly_salary"],no_car["num_children"],color="red",label="no Car")
    plt.xlabel("Monthly salary in $")
    plt.ylabel("Number of children")
    plt.title("monthly salary and number of children")
    plt.legend()
    plt.show()


def calculate_Euclidean(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def predict(x2,y2,k):  
    if(data.shape[0] < k):
        return "that's too large number of neighbours"

    all_distances = []
    user_has_car = 0
    user_dont_have_car = 0

    for i, row in data.iterrows():
        x1 = row["monthly_salary"]
        y1 = row["num_children"]
        dist = calculate_Euclidean(x1, y1, x2, y2)
        all_distances.append((dist,row))
        
    all_distances.sort(key=lambda x:x[0])

    for i in range(k):
        dist, row = all_distances[i]
        if (row["has_car"] == 1):
            user_has_car += 1
        else:
            user_dont_have_car += 1

    if(user_has_car > user_dont_have_car):
        return "You have a car !!!"
    else:
        return "You dont have a car !!!"


visualize()

x2 = float(input("Enter monthly salary $(1000-100000): "))
y2 = float(input("Enter number of children (0-5): "))
k = int(input("Enter the number of neighbours: "))
result = predict(x2,y2,k)

if result :
    print(result)
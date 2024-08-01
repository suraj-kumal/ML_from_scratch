import pandas as pd
import numpy as np

num_samples = 50 


np.random.seed(42)


num_children = np.random.randint(0, 6, size=num_samples)


monthly_salary = np.random.randint(1000, 100000, size=num_samples)

has_car = np.random.randint(0, 2, size=num_samples)


df = pd.DataFrame({
    'num_children': num_children,
    'monthly_salary': monthly_salary,
    'has_car': has_car
})

csv_filename = 'dataset.csv'
df.to_csv(csv_filename, index=False)

print(f"Dataset with {num_samples} samples has been generated and saved to '{csv_filename}'.")

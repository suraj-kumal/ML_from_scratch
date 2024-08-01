import pandas as pd
import numpy as np

np.random.seed(0)

n_samples = 100

studytime = np.random.randint(1, 11, size=n_samples)

base_score = 50
noise = np.random.normal(0, 10, size=n_samples)

score = base_score + 5 * studytime + noise

score = np.clip(score, a_min=None, a_max=100)

data = pd.DataFrame({
    'studytime': studytime,
    'score': score
})

data.to_csv('studytime_score_dataset.csv', index=False)

print("Dataset created and saved as 'studytime_score_dataset.csv'.")

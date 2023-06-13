import pandas as pd

existing_data = pd.read_csv('Twitter_Data.csv')

new_data = pd.read_csv('Reddit_Data.csv')

combined_data = pd.concat([existing_data, new_data], ignore_index=True)

shuffled_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

shuffled_data.to_csv('shuffled_data.csv', index=False)
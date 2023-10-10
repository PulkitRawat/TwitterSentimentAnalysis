import pandas as pd

sentiment = ['positive', 'neutral', 'negative']
df = pd.read_csv('Twitter_Data.csv')
print(df.head())
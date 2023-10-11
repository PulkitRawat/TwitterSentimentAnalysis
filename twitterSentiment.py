import pandas as pd

sentiment = ['positive', 'neutral', 'negative']
df = pd.read_csv('Twitter_Data.csv')
# print(df.head())

df = df.drop_duplicates()
df = df.dropna(subset=['clean_text', 'category'])

# # print(df.head())
# duplicates = df.duplicated()
# df_duplicate = df[duplicates]
# print('duplicate rows')
# print(df_duplicate)

# print('missing values rows')
# missing_values = df.isna()
# rows_with_missing = missing_values.any(axis=1)
# print(df[rows_with_missing])    
import pandas as pd
import nltk

from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')

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

def preprocessing(text:str):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    text = ' '.join(tokens)
    return text

df['clean_text'] = df['clean_text'].apply(preprocessing)
print(df.head())

# Split the dataset into training, validation, and test sets
train_df, val_test_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)





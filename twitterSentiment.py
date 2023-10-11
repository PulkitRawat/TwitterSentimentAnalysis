import pandas as pd
import nltk

from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer, TFBertForSequenceClassification 

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
    tokenized = train_df['clean_text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=128, truncation=True)))
    return tokenized


train_df, val_test_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)

model_name = 'bert-base-uncased'  
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name)

train_df['clean_text'] = train_df['clean_text'].apply(preprocessing)
print(train_df.head())




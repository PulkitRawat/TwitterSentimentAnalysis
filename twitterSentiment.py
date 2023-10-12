import pandas as pd
import nltk

from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# nltk.download('stopwords')
# nltk.download('punkt')

sentiment = ['positive', 'neutral', 'negative']
df = pd.read_csv('Twitter_Data.csv')
# print(df.head())

df = df.drop_duplicates()
df = df.dropna(subset=['clean_text', 'category'])

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

train_df, val_test_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=10000)

X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['clean_text'])
X_val_tfidf = tfidf_vectorizer.transform(val_df['clean_text'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['clean_text'])

model = LogisticRegression()
model.fit(X_train_tfidf, train_df['category'])

val_predictions = model.predict(X_val_tfidf)
accuracy = accuracy_score(val_df['category'], val_predictions)
print(f"Validation Accuracy: {accuracy:.2f}")
print(classification_report(val_df['category'], val_predictions))



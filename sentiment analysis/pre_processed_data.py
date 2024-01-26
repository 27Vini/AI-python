import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('stopwords')

columns = ["polarity", "tweet_id", "date", "query", "user", "text"]
df = pd.read_csv("sentiment140.csv", encoding="ISO-8859-1",names=columns,header=None)
#print(df)

df = df.drop(["date","tweet_id","query","user"],axis=1)
#print(df)

df["polarity"] = df["polarity"].map({0:0,2:1,4:2})
df = df.drop_duplicates(subset=["text"])
#print(df)

def formating_and_tokenizing(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()

    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]


    return " ".join(tokens)

df["processed_text"] = df["text"].apply(formating_and_tokenizing)
df.to_csv("sentiment140_processed.csv",index=False)
print(df.head())
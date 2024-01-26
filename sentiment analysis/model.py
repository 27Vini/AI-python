import pandas as pd


df = pd.read_csv("sentiment140_processed.csv", encoding="ISO-8859-1")


print(df.loc[df["polarity"] ==2, ['polarity','text','processed_text']])

import pandas as pd

# Sample DataFrame
df = pd.read_csv('data.csv')

df['Sentiment'] = df['Sentiment'].replace('positive', 1)
df['Sentiment'] = df['Sentiment'].replace('negative', 0)
df['Sentiment'] = df['Sentiment'].replace('neutral', 0.5)


#df.to_csv('sentement_data.csv', index=False)
print(df.head(10))
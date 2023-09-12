
import pandas as pd

# Sample DataFrame
data = pd.read_csv('data.csv')

df = pd.DataFrame(data)

# Replace 'positive' with 1 in the 'Sentiment' column
df['Sentiment'] = df['Sentiment'].replace('positive', 1)
df['Sentiment'] = df['Sentiment'].replace('negative', 0)
df['Sentiment'] = df['Sentiment'].replace('neutral', 0.5)
# Display the modified DataFrame
#df.to_csv('sentement_data.csv', index=False)

print(df.head(10))
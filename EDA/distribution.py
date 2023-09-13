
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


df = pd.read_csv('data.csv')

def ternary_pie_chart(df):
    sentiment = df['Sentiment'].value_counts().tolist()
    labels = ['Positive', 'Negative', 'Neutral']
    colors = ['#5DE31D', 'red', 'grey']
    plt.pie(sentiment, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title('Sentiment Distribution')
    plt.show()
#ternary_pie_chart(df)

def binary_pie_chart(df):
    df['Sentiment'] = df['Sentiment'].replace('neutral', 'negative')
    sentiment = df['Sentiment'].value_counts().tolist()
    print(sentiment)
    labels = ['Positive', 'Negative']
    colors = ['#5DE31D', 'red']
    plt.pie(sentiment, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title('Sentiment Distribution')
    plt.show()
#binary_pie_chart(df)



# Count the number of occurrences for each sentiment
sentiment_counts = df['Sentiment'].value_counts()

value_counts = df['Sentiment'].value_counts()

# Create a DataFrame or Series with labels
value_counts_with_labels = pd.DataFrame({'Category': value_counts.index, 'Count': value_counts.values})

print(value_counts_with_labels)
# You can also reset the index if you want to make 'Category' a regular column
# value_counts_with_labels = value_counts_with_labels.reset_index(drop=True)

print(value_counts_with_labels)
print(sentiment_counts)
# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the pie chart
plt.show()

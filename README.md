# Financial-Sentiment-Analysis
Sentiment analysis of financial texts, Python Notebooks of each model are included in the [classification](https://github.com/EliBrignac/Financial-Sentiment-Analysis/tree/main/classification) folder in this repository.

### Table of Contents
- [Financial-Sentiment-Analysis](#financial-sentiment-analysis)
    - [Table of Contents](#table-of-contents)
  - [Summary](#summary)
      - [Figure1:](#figure1)
  - [Data Examples](#data-examples)
  - [Ternary Classification Model](#ternary-classification-model)
  - [Binary Classification Model](#binary-classification-model)


## Summary
In this project, I made 2 models to predict the sentiment of financial texts. The first model is a binary classification model that has 92% accuracy, and the second model is a ternary classification model with 78% accuracy. The models were both trained on a kaggle dataset with nearly 6000 entries, the dataset can be found [HERE](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis). The models I made were created by fine-tuning a basic bert model that is part of the `transformers` library. This method of text classification proved to be superior to other methods of text classification that I found in other kaggle code submissions [Figure1](#figure1). The other methods failed to cross the 70% accuracy threshold, while both of my models acheived accuracies far beyond that mark. Both of my models were trained using an A100 Google Colab GPU.

#### Figure1:
###### Here is a table of other methods of classification that I found in another kaggle submission, compared to my methods. All other models found were for ternary classification (source [HERE](https://www.kaggle.com/code/chibss/financial-sentiment-analysis)).


| Their Models                       | Accuracy    | My Models                       | Accuracy    |
| ----------------------------| ----------- | ----------------------------| ----------- |
| CatBoostClassifier           | <font color="red">66.89%</font>      | Binary Classification Model | <font color="green">92.50%</font>         |
| AdaBoostClassifier           | <font color="red">65.61%</font>      | Ternary Classification Model| <font color="green">78.63%</font>         |
| SGDClassifier                | <font color="red">63.99%</font>       |                             |             |
| RandomForestClassifier       | <font color="red">62.02%</font>       |                             |             |
| DecisionTreeClassifier       | <font color="red">59.11%</font>       |                             |             |
| KNeighborsClassifier         | <font color="red">54.32%</font>       |                             |             |
| GaussianNB                   | <font color="red">53.38%</font>       |                             |             |
| LogisticRegression           | <font color="red">52.35%</font>       |                             |             |

## Data Examples

| Text                                                                                                      | Sentiment  |
|-----------------------------------------------------------------------------------------------------------|------------|
| "The GeoSolutions technology will leverage Benefon's GPS solutions by providing Location Based Search Technology, a Communities Platform, location relevant multimedia content and a new and powerful commercial model." | <span style="color:green">positive</span>   |
| "$ESI on lows, down $1.50 to $2.50 BK a real possibility"                                                  | <span style="color:red">negative</span>   |
| "For the last quarter of 2010, Componenta's net sales doubled to EUR131m from EUR76m for the same period a year earlier, while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m." | <span style="color:green">positive</span>   |
| "According to the Finnish-Russian Chamber of Commerce, all the major construction companies of Finland are operating in Russia." | <span style="color:yellow">neutral</span>    |
| "The Swedish buyout firm has sold its remaining 22.4 percent stake, almost eighteen months after taking the company public in Finland." | <span style="color:yellow">neutral</span>    |
| "$SPY wouldn't be surprised to see a green close."                                                        | <span style="color:green">positive</span>   |
| "Shell's $70 Billion BG Deal Meets Shareholder Skepticism."                                               | <span style="color:red">negative</span>   |

## Ternary Classification Model 
The ternary classification model acheived an accuracy of 78% and classified the financial texts as either positive, negative, or neutral. The distribution of the dataset was 53.6% neutral, 31.7% positive, and 14.7% negative A report of the model is below.
```
              precision    recall  f1-score   support

    negative       0.53      0.48      0.51        50
     neutral       0.80      0.83      0.82       149
    positive       0.88      0.87      0.88        94

    accuracy                           0.78       293
   macro avg       0.74      0.73      0.73       293
weighted avg       0.78      0.78      0.78       293
```
The lower than expected percision metric is likely due to the fact that the model was trained on a dataset that was not balanced. This means that the model was trained on a dataset where some categories were under represented. Notice that the dataset had a 2:1 ratio of neutral to positive values, and a 4:1 ratio of neutral to negative values. This is likely the reason for the lower than expected negative precision metric. To improve this, we could either make it a binary classifier (like I did) or we could balance the dataset by adding more negative and positive values. Note that the later solution would require us to add more data to the dataset, which is not always possible.

## Binary Classification Model
The binary classification model acheived a very high accuracy of 92% and classified the financial texts as either positive or negative. All texts that were originally labled as neutral, were re-labled to be negative. This gave us a positive-negative split of approximately 68.3% negative values and 31.7% positive values. A report of the model is below.


```
              precision    recall  f1-score   support

    negative       0.94      0.93      0.94       199
    positive       0.86      0.88      0.87        94

    accuracy                           0.92       293
   macro avg       0.90      0.91      0.91       293
weighted avg       0.92      0.92      0.92       293
```
This model is very good there isn't much else to say 🙂

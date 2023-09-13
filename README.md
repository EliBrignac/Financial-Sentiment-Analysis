# Financial-Sentiment-Analysis
Sentiment Analysis of Financial Texts

## Summary
In this project, I made 2 models to predict the sentiment of financial texts. The first model is a binary classification model that has 92% accuracy, and the second model is a ternary classification model with 78% accuracy. The models were both trained on a kaggle dataset with nearly 6000 entries, the dataset can be found [HERE](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis).

## How the models were made
The models were made by fine-tuning a basic bert model that is part of the `transformers` library. This method of text classification proved to be superior to other methods of text classification that I found in other kaggle code submissions (Figure1). The other methods failed to cross the 70% accuracy threshold, while both of my models acheived accuracies way past that mark.

###### [Figure1](https://www.kaggle.com/code/chibss/financial-sentiment-analysis) (source linked)
| Model | Accuracy |
| --- | --- |
|  CatBoostClassifier | 66.894782 |
|  AdaBoostClassifier | 65.611634 |
| SGDClassifier | 63.986313 | 
|  RandomForestClassifier | 62.018820 | 
|  DecisionTreeClassifier | 59.110351 | 
| KNeighborsClassifier | 54.319932 |
|  GaussianNB | 53.378956 | 
| LogisticRegression | 52.352438 |


## Binary Classification Model
The binary classification model acheived a very high accuracy of 92% and classified the financial texts as either positive or negative. All texts that were originally labled as neutral, were re-labled to be negative. This gave us a positive-negative split of [SPLIT HERE]. A report of the model is below


```
              precision    recall  f1-score   support

    negative       0.94      0.93      0.94       199
    positive       0.86      0.88      0.87        94

    accuracy                           0.92       293
   macro avg       0.90      0.91      0.91       293
weighted avg       0.92      0.92      0.92       293
```

## Ternary Model 
The ternary classification model acheived an accuracy of 78% and classified the financial texts as either positive, negative, or neutral. A report of the model is below
```
              precision    recall  f1-score   support

    negative       0.53      0.48      0.51        50
     neutral       0.80      0.83      0.82       149
    positive       0.88      0.87      0.88        94

    accuracy                           0.78       293
   macro avg       0.74      0.73      0.73       293
weighted avg       0.78      0.78      0.78       293

```

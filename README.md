# Data Science with Python Class

# Machine Learning Classification Project

The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).

The bank-marketing.csv dataset contains 41188 observations (rows) and 21 features (columns). The dataset contains 10 numerical features (age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed) and 11 nominal features (job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome, y) that were converted into factors with numerical value designated for each level. Dataset available on https://www.kaggle.com/henriqueyamahata/bank-marketing

Backward elimination is used as feature elimination technique.

K-fold cross validation was used to evaluate the success of machine learning models.

Classification Techniques:
1) Logistic Regression
2) Decision Tree Classifier
3) Random Forest Classifier
4) K – Nearest Neighbor
5) Support Vector Machine

Measuring the Performance of Classification Models:
1) Confussion Matrix
2) Accuracy
3) Error Rate / Misclassification Rate

The algorithms was implemented using Python and packages of pandas and sklearn. Matplotlib and seaborn was used for plotting the data. 

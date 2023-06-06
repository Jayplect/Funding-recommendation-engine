## Overview of the Analysis

Building a model that can accurately identify the creditworthiness of borrowers is crucial for P2P lending services. Such a model allows lenders to evaluate the risk associated with lending to a particular borrower, set appropriate interest rates, and make well-informed decisions about loan approvals. By leveraging historical borrower data and various financial indicators, lenders can assess the probability of default and make more reliable lending decisions. In this project, I used a dataset containing the historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. The dataset contained features such as loan size, interest rate, borrower income, debt to income ratio, number of account the borrower posseses, a count of derogatory remarks, as well as their total debt while the loan status formed the label.

## Dependencies used

![SkLearn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)

## ML Model Design Process
The ML model design process (Fig 1) began with ingesting the labels into a model to split the data into training and testing sets. To determine the loan lending creditworthiness of an individual I used a logistic regression. To futher improve the balanced score observed from the logistic model, I resampled the training data using a random oversampling model classifier. 

Fig 1: Model design process used for this study

![image](https://github.com/Jayplect/credit-risk-classification/assets/107348074/abdb6ff2-6370-4b1b-b2ce-3664fc54be89)

## Results

#### Description of Model Accuracy, Precision, and Recall scores.

- From the confusion matrix (Fig. 2) the logistic model predicted 18,663 as healthy loans correctly and 102 as healthy loans incorrectly from a total loan status of 18,765 which were healthy (i.e., low-risk).
- The model predicted 563 as high-risk loans correctly and 56 high-risk loans incorrectly from a total loan status of 619 which were high-risk.

> Fig 2: Confusion Matrix showing the Imbalance proportions of the actual and predicted classes

![image](https://github.com/Jayplect/credit-risk-classification/assets/107348074/57e56a59-e561-4510-8d5d-60a5ad3939f6)

- The classification report which takes into account the model's accuracy revealed that the healthy loans had a precision score of 100%, and a recall of 99%. For the non-healthy loans, the precision and recall were 85% and 99%, respectively. The overall performance of the model was calculated to be 99%.

#### Description of Model Accuracy, Precision, and Recall scores after Resampling:

- From the confusion matrix (Fig. 3) the logistic model predicted 18,663 as healthy correctly and 102 as healthy incorrectly from a total loan status of 18,765 which were healthy (i.e., low-risk).
- The model predicted 563 as high-risk loans correctly and 56 high-risk loans incorrectly from a total loan status of 619 which were high-risk.

> Fig 3: Confusion Matrix showing the Imbalance proportions of the actual and predicted classes after resampling

![image](https://github.com/Jayplect/credit-risk-classification/assets/107348074/b44648fc-4bfa-40fa-b112-262111429b49)

- After resampling using the oversampling method, the classification report showed that the healthy loans had a precision score of 100%, and a recall of 99%. For the non-healthy loans, the precision and recall were 84% and 99%, respectively. The overall performance of the model was calculated to be 99%.

## Discussion
- The logistic regression model performs well according to the balanced accuracy score of approximately 95%. However from the classification report, for the Risky loans, we get a relatively smaller recall, and precision is mixed compared to the healthy loans with a precision of 100%. Because the Healthy Loan class is so much larger, the classifier focuses on getting that class(the healthy loan) right, and not the smaller high-risky Loan class. The physical meaning of this is that lenders refusing to provide loans classed as high-risk loans would be 15% of the time adjudged to be healthy loans.
- Oversampling improves the balanced score from 95% to 99%, as observed randomly  oversampling and fitting the training datasets using the logistic regression model. The imporvement in the scoring metrics is because the oversampling algorithm allows for selection of random samples from the minority class (in this case the High-Risk Loans) with replacement and supplementing the training data with multiple copies of this instance. The issue with such kind of sampling is that it may overrepresent a single sample in the training set. However this might not be the case for our dataset because of the high disparity between high-risk and healthy loan classes as observed in the confusion model. 

## Summary
- Overall, the accuracy seems to be good enough to start exploring this kind of algorithms in a bank, however, I would prefer to start running a pilot with new data to assess model's reliability.
- It's important to note that the effectiveness of the creditworthiness model heavily depends on the quality and relevance of the data collected. Therefore, it's crucial to have a comprehensive and reliable dataset to achieve accurate predictions.
- As part of next steps, it will be imperative to identify the most relevant features that are likely to impact the creditworthiness of borrowers and then run a pilot with new data sets of data to assess model's reliability.

## References
Data for this dataset was generated by edX Boot Camps LLC, and is intended for educational purposes only.

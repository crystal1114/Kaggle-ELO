# Kaggle-ELO
Kaggle competition - Elo Merchant Category Recommendation.

Developed algorithms by using Python to identify and serve the most relevant opportunities to individuals.
Predicted a loyalty score for each card_id by uncovering signals in customer characteristics and behaviors.
https://www.kaggle.com/c/elo-merchant-category-recommendation



1) Overview
Identify and serve the most relevant opportunities to individuals, by uncovering signal in customer loyalty. Your input will improve customers’ lives and help Elo reduce unwanted campaigns, to create the right experience for customers.


2) Dataset
Historical_transactions.csv contains up to 3 months' worth of transactions for every card at any of the provided merchant_ids. new_merchant_transactions.csv contains the transactions at new merchants (merchant_ids that this particular card_id has not yet visited) over a period of two months.
merchants.csv contains aggregate information for each merchant_id represented in the data set.

3) File descriptions

• train.csv - the training set
• test.csv - the test set
• sample_submission.csv - a sample submission file in the correct format - contains all card_ids you are expected to predict for.
• historical_transactions.csv - up to 3 months' worth of historical transactions for each card_id
• merchants.csv - additional information about all merchants / merchant_ids in the dataset.
new_merchant_transactions.csv - two months' worth of data for each card_id containing ALL purchases that card_id made at merchant_ids that were not visited in the historical data.

4) Evaluation
Submissions are scored on the RMSE(Root Mean Squared Error).

5) A summary of Analyze Process
  a. EDA
  
    i. Data Collection
    Integration
    ii. Visualization
    Grasp difficult concepts or identify new patterns.
    iii. Data Cleaning
    The primary goal of data cleaning is to detect and remove errors and anomalies to increase the value of data in analytics and decision making。
    These include missing value imputation, outliers detection, transformations, integrity constraints violations detection and repair, consistent query answering, deduplication, and many other related problems such as profiling and constraints mining.
    iv. Data Preprocessing
      - removing Target column (id)
      - Sampling (without replacement)
      - Making part of iris unbalanced and balancing (with undersampling and SMOTE)
      - Introducing missing values and treating them (replacing by average values)
      - Noise filtering
      - Data discretization
      - Normalization and standardization
      - PCA analysis
      - Feature selection (filter, embedded, wrapper)

  b. Feature Engineering

  c. Machine Learning model
  
    i. Categorize by input:
      - If have labelled data, it’s a supervised learning problem.
      - If have unlabelled data and want to find structure, it’s an unsupervised learning problem.
      - If want to optimize an objective function by interacting with an environment, it’s a reinforcement learning problem. 
    ii. Categorize by output.
      - If the output of model is a number, it’s a regression problem.
      - If the output of model is a class, it’s a classification problem.
      - If the output of model is a set of input groups, it’s a clustering problem.
    iii. Understand your constraints
      - Data storage capacity. Depending on the storage capacity of system, might not be able to store gigabytes of classification/regression models or gigabytes of data to clusterize. This is the case, for instance, for embedded systems.
      - Does the prediction have to be fast? In real time applications, it is obviously very important to have a prediction as fast as possible. For instance, in autonomous driving, it’s important that the classification of road signs be as fast as possible to avoid accidents.
      - Does the learning have to be fast? In some circumstances, training models quickly is necessary: sometimes, need to rapidly update, on the fly, your model with a different dataset.
    iv. Find the available algorithms
      - Whether the model meets the business goals
      - How much pre processing the model needs
      - How accurate the model is
      - How explainable the model is
      - How fast the model is: How long does it take to build a model, and how long does the model take to make predictions.
      - How scalable the model is

  d. Model choosen, and Advantages of Light GBM
  
      i. Faster training speed and higher efficiency: Light GBM use histogram based algorithm i.e it buckets continuous feature values into discrete bins which fasten the training procedure.
      ii. Lower memory usage: Replaces continuous values to discrete bins which result in lower memory usage.
      iii. Better accuracy than any other boosting algorithm: It produces much more complex trees by following leaf wise split approach rather than a level-wise approach which is the main factor in achieving higher accuracy. However, it can sometimes lead to overfitting which can be avoided by setting the max_depth parameter.
      iv. Compatibility with Large Datasets: It is capable of performing equally good with large datasets with a significant reduction in training time as compared to XGBOOST.
    Parallel learning supported.


6) Further work
A linear or a logistic modeling technique is great and does well in a variety of problems but the drawback is that the model only learns the effect of all variables or features individually rather than in combination. 
So we try to use Field-Aware Factorization Machines method to tackle this problem.




7) Families of ML algorithms
There are several categories for machine learning algorithms, below are some of these categories:

• Linear
• Linear Regression
• Logistic Regression
• Support Vector Machines
• Tree-Based
• Decision Tree
• Random Forest
• GBDT
• KNN
• Neural Networks

And if we want to categorize ML algorithms with the type of learning, there are below type:

• Classification
• k-Nearest Neighbors
• LinearRegression
• SVM
• DT 
• NN
• clustering
• K-means
• HCA
• Expectation Maximization
• Visualization and dimensionality reduction:
• Principal Component Analysis(PCA)
• Kernel PCA
• Locally -Linear Embedding (LLE)
• t-distributed Stochastic Neighbor Embedding (t-SNE)
• Association rule learning
• Apriori
• Eclat
• Semisupervised learning
• Reinforcement Learning
• Q-learning
• Batch learning & Online learning
• Ensemble Learning

  




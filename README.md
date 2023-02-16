# ML-program-to-diagnose-breast-cancer
• First, I made sure I had imported all the necessary libraries. To continue, I used the load breast cancer() function to access the predefined breast cancer dataset. X and Y were both taken into account as potential input features and output variables.
• In the lines of code that followed, I attempted to divide my data set into training and testing data, allocating 20% of the total to the latter.
• I utilized the in-built logistic Regression() function to train a logistic regression model on the training data set.
• I used GridSearchCV() to locate the best parameters among the defined hyperparameters and use them to fit the data.
• The best model that best fits the testing data is determined by running best.model predict on the testing data set. To determine the best model's precision, recall, and false-positive rate, we used the confusion matrix and accuracy score.


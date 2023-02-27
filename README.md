# Sentiment Analysis of Tweets using Logistic Regression

The code is a Python script that performs sentiment analysis on textual data. The script uses the Pandas library to load and preprocess the training and test data, and the NLTK library to tokenize the text and remove stop words. It also uses the Scikit-learn library to convert the text data into numerical features using a bag of words model, and to train a logistic regression model to predict the sentiment of the text.

The script begins by loading the training and test data from CSV files. It then concatenates the two datasets and preprocesses the text data by converting it to lowercase, removing punctuation, and removing numbers. It then tokenizes the text and removes stop words, and joins the tokenized text back into a single string. The script then converts the text data into numerical features using the CountVectorizer function, which creates a bag of words model with a maximum of 5000 features. The data is split into training and testing sets using the train_test_split function from Scikit-learn.

The script then initializes and fits a StandardScaler object to the training data, which scales the data by subtracting the mean and dividing by the standard deviation. The training and test data are then transformed using the fitted scaler. A logistic regression model is initialized and trained on the scaled training data. The script then uses the trained model to make predictions on the test data and calculates the accuracy of the model using the accuracy_score function from Scikit-learn. It also outputs a classification report that includes precision, recall, and F1-score metrics.

The script saves the trained model to disk using the pickle library, which serializes the model object and writes it to a file. The saved model can then be loaded using the pickle.load function.

Finally, the script loads new data from two CSV files and preprocesses it in the same way as the training data. It then uses the trained model to make predictions on the new data, and outputs the number of positive and negative predictions.

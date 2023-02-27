import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


stop_words = set(stopwords.words("english"))


# Load the training data
training_data = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding = "ISO-8859-1", header=None)
training_data.columns = ["sentiment", "id", "date", "query", "user", "text"]

# Load the test data
test_data = pd.read_csv("testdata.manual.2009.06.14.csv", encoding = "ISO-8859-1", header=None)
test_data.columns = ["sentiment", "id", "date", "query", "user", "text"]

# Combine the training and test data
data = pd.concat([training_data, test_data])

# Preprocess the data
data['text'] = data['text'].str.lower() # Convert text to lowercase
data['text'] = data['text'].str.replace('[^\w\s]','') # Remove punctuation
data['text'] = data['text'].str.replace('\d+', '') # Remove numbers

# Output the preprocessed data
#print(data.head())


# Tokenize the text
data['text'] = data['text'].apply(word_tokenize)

# Output the tokenized data
#print(data.head())

# Remove stop words
stop_words = set(stopwords.words("english"))
data['text'] = data['text'].apply(lambda x: [word for word in x if word not in stop_words])

# Join the tokenized text back into a single string
data['text'] = data['text'].apply(lambda x: " ".join(x))

# Convert the text data into numerical features using bag of words
vectorizer = CountVectorizer(max_features=5000)
data_features = vectorizer.fit_transform(data['text'])

# Output the numerical features
#print(data_features.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_features, data['sentiment'], test_size=0.20, random_state=42)

# Initialize the scaler
scaler = StandardScaler(with_mean=False)

# Fit the scaler to the training data
scaler.fit(X_train)

# Transform the training and test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Evaluate the model performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model to disk
filename = 'final_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Load the saved model
loaded_model = pickle.load(open(filename, 'rb'))

# Use the loaded model to make predictions on new data
new_data_train = pd.read_csv("Corona_NLP_train.csv", encoding = "ISO-8859-1", header=None)
new_data_train.columns = ["sentiment", "id", "date", "query", "user", "text"]

new_test_train = pd.read_csv("Corona_NLP_test.csv", encoding = "ISO-8859-1", header=None)
new_test_train.columns = ["sentiment", "id", "date", "query", "user", "text"]

# Combine the training and test data
new_data = pd.concat([new_data_train, new_test_train])

# Preprocess the new data in the same way as you preprocessed the training data
new_data['text'] = new_data['text'].str.lower() # Convert text to lowercase
new_data['text'] = new_data['text'].str.replace('[^\w\s]','') # Remove punctuation
new_data['text'] = new_data['text'].str.replace('\d+', '') # Remove numbers

# Tokenize the new text data
new_data['text'] = new_data['text'].apply(word_tokenize)

# Remove stop words from the new data
stop_words = set(stopwords.words("english"))
new_data['text'] = new_data['text'].apply(lambda x: [word for word in x if word not in stop_words])

# Join the tokenized new text back into a single string
new_data['text'] = new_data['text'].apply(lambda x: " ".join(x))

# Convert the new text data into numerical features using the same bag of words model
new_data_features = vectorizer.transform(new_data['text'])

# Scale the new numerical features in the same way as you scaled the training data
new_data_features = scaler.transform(new_data_features)

# Use the loaded model to make predictions on the new data
new_data_predictions = loaded_model.predict(new_data_features)
print(new_data_predictions)

# Count the number of positive and negative predictions
positive_predictions = sum(1 for prediction in new_data_predictions if prediction == 4)
negative_predictions = sum(1 for prediction in new_data_predictions if prediction == 0)

# Output the counts
print("Positive Predictions:", positive_predictions)
print("Negative Predictions:", negative_predictions)

# Get the average sentiment
average_sentiment = sum(new_data_predictions) / len(new_data_predictions)

# Output the average sentiment
print("Average Sentiment:", average_sentiment)

new_data['sentiment_prediction'] = new_data_predictions
new_data.to_csv("predictions.csv", index=False)

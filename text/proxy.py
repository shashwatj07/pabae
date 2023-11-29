# Importing necessary libraries
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load the TREC 2005 spam dataset
with open('/scratch/sharma96/ab/tasti/data/trec05p/trec05p-1/spam25/index', 'r', encoding='utf-8') as idxf:
    trec_labels = pd.read_table(idxf, sep=' ', names=['label', 'path'])
print(trec_labels.head())

trec_labels['name'] = trec_labels['path'].str.replace(r'^\.\./data/', '')
trec_labels.head()

trec_mails = {}
for name in tqdm(trec_labels['name']):
    path = f'trec07p/data/{name}'
    with trec_zf.open(path) as mailf:
        content = mailf.read()
        content = content.decode('latin1')
        content = unicodedata.normalize('NFKD', content)
        trec_mails[name] = content
trec_mails = pd.Series(trec_mails, name='content')
len(trec_mails)

# Extracting features and labels from the dataset
data = []
labels = []
for email in emails:
    if email.startswith('ham'):
        labels.append(0)  # 0 for non-spam (ham) emails
        data.append(email[4:])
    elif email.startswith('spam'):
        labels.append(1)  # 1 for spam emails
        data.append(email[5:])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Vectorize the email text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Build and train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test_vectorized)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

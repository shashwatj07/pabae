import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import gensim.downloader as api
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset (replace 'path/to/office_supplies_reviews.csv' with the actual path)
input_file = '/scratch/aa117/data/amazon/amazon_office_df_processed.csv'
df = pd.read_csv(input_file)

# Drop NaN values
# df = df.dropna(subset=['processed_review', 'rating'])

# Tokenize the reviews
# stop_words = set(stopwords.words('english'))
# df['tokens'] = df['review'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words])

# Load pre-trained GloVe embeddings
glove_model = api.load("glove-wiki-gigaword-100")

# Map words to their GloVe embeddings
def get_embedding(word):
    try:
        return torch.tensor(glove_model[word])
    except KeyError:
        # If the word is not in the vocabulary, return a zero vector
        return torch.zeros(100)

tqdm.pandas()
df['embeddings'] = df['processed_review'].progress_apply(lambda x: [get_embedding(word) for word in x])
print("Embeddings created.")
del df['processed_review']
#save df as pickle
df.to_pickle('/scratch/aa117/data/amazon/amazon_office_df_processed_embeddings.pkl')

# Define a simple neural network model
class RatingPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(RatingPredictor, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1, :, :])

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Encode labels to integers
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['rating'])
test_labels = label_encoder.transform(test_df['rating'])

# Create DataLoader for training
class OfficeSuppliesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

embedding_dim = 100
hidden_dim = 128
output_dim = 5  # Ratings are on a scale of 1-5

train_dataset = OfficeSuppliesDataset(train_df['embeddings'].values, train_labels)
print("Dataset created.")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print("DataLoader created.")

# Initialize the model, loss function, and optimizer
model = RatingPredictor(embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    predictions = []
    true_labels = []

    for embeddings, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        embeddings = torch.stack(embeddings)
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions.extend(torch.argmax(outputs, dim=1).tolist())
        true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

# Evaluation on the test set
model.eval()
test_embeddings = torch.stack(test_df['embeddings'].values)
test_outputs = model(test_embeddings)
test_predictions = torch.argmax(test_outputs, dim=1).tolist()

test_accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")

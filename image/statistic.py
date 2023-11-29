import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
# from deepface import DeepFace
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# Define the path to your CSV file and the root directory containing the images
img_dir = '/scratch/aa117/osprey/proxycode/amazon/posters'
# output_dir = '/scratch/aa117/osprey/proxycode/amazon'
output_dir = "/scratch/aa117/data/all_proxies/amazon_posters"
os.makedirs(output_dir, exist_ok=True)

# Path to the original CSV file
original_csv_path = '/scratch/aa117/osprey/proxycode/amazon/predicate_statistic.csv'

# Path to the new CSV file with sampled rows
csv_file = os.path.join(output_dir, 'sampled-amazon-posters-statistic.csv')

# Specify the path where you want to save the weights
weights_path = os.path.join(output_dir, 'amazon-statistic4.pth')
override = False


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = int(max(self.df.iloc[:, 2])) - int(min(self.df.iloc[:, 2])) + 1
        print("Classes: {}".format(self.classes))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        label = torch.nn.functional.one_hot(torch.tensor(int(self.df.iloc[idx, 2]))-1, self.classes)
        name = self.df.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)

        return image, label, name


# Number of rows to randomly sample
# num_samples = 1000

# Load the original CSV file into a DataFrame
original_df = pd.read_csv(original_csv_path)#, header=None)
# original_df.columns=['image', 'label']

# Separate data into True and False classes
true_class = original_df[original_df['label'] == 1]
false_class = original_df[original_df['label'] == 0]

# Sample an equal number of images from both classes
# num_samples = min(len(true_class), len(false_class))
# print("Sampling {} images from each class".format(num_samples))

# Create a new DataFrame using equal samples of each rating from true_class
num_samples = len(true_class)
for i in range(1, 6):
    true_class_i = true_class[true_class['statistic'] == i]
    print("Class {}: {}".format(i, len(true_class_i)))
    num_samples = min(len(true_class_i), num_samples)
    
subset_list = []
for i in range(1, 6):
    true_class_i = true_class[true_class['statistic'] == i]
    subset_list.append(true_class_i.sample(n=1024+128*i, random_state=42))
subset_df = pd.concat(subset_list)

# Shuffle the subset DataFrame
subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the subset DataFrame to a new CSV file
subset_df.to_csv(csv_file, index=False)

# Randomly sample 1000 rows
# sampled_df = original_df.sample(n=num_samples)  # Set a random seed for reproducibility

# Save the sampled DataFrame to a new CSV file
# sampled_df.to_csv(csv_file, index=False)

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the dataset
dataset = CustomDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
# print(dataset)
# print(dataset[0])

# Create a DataLoader for batching and shuffling (optional)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for data, labels, name in dataloader:
    print(data.size(), len(labels), name)
    break



# Load pre-trained MobileNetV2 model
# model = models.mobilenet_v2(pretrained=True)

# Modify the final fully connected layer to match the number of classes in your dataset
# num_classes = 2
# model.classifier[1] = nn.Linear(model.last_channel, num_classes)

class FineTunedMobileNetV2(nn.Module):
    def __init__(self, num_classes_regression):
        super(FineTunedMobileNetV2, self).__init__()

        # Load pre-trained MobileNetV2 model
        self.mobilenet_v2 = models.mobilenet_v2(weights='DEFAULT')

        # Freeze weights of the old parameters
        for param in self.mobilenet_v2.parameters():
            param.requires_grad = False

        # Define regression head
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 256),  # Adjust input features based on the output of MobileNetV2
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_regression),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Feature extraction using MobileNetV2
        features = self.mobilenet_v2.features(x)
        
        # Global average pooling
        # features = self.mobilenet_v2.avgpool(features)
        # features = torch.flatten(features, 1)
        
        # Regression head
        regression_output = self.regression_head(features)

        return regression_output

# Example usage:
num_classes_regression = dataset.classes  # Modify based on your regression task
model = FineTunedMobileNetV2(num_classes_regression)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=450, verbose=True)
# Training loop
num_epochs = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))
model.to(device)

if (os.path.exists(weights_path) and not override) or num_epochs == 0:
    print("Loading weights from {}".format(weights_path))
    model.load_state_dict(torch.load(weights_path))
else:
    print("Training from scratch")
    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        # all_predictions = []
        # all_labels = []
        for i, data in enumerate(tqdm(dataloader)):
            inputs, labels, _ = data
            # print(inputs.shape)
            inputs, labels = inputs.to(device), labels.to(device)
            label_class = torch.argmax(labels, dim=1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward, backward, and optimize
            outputs = model(inputs)
            predicted_class = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(predicted_class == label_class)
            total_samples += len(label_class)

            # Assuming binary classification, adjust labels accordingly
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Update the learning rate scheduler based on the loss
            scheduler.step(loss)
            # Print statistics
            running_loss += loss.item()

            if (i+1)%100 == 0:
                print('[Epoch: %d, Step: %5d] Loss: %.3f Accuracy: %.3f' % (epoch+1, i+1, running_loss/(i+1), correct_predictions/total_samples))

        print('[Epoch: %d] Softmax Loss: %.3f Accuracy: %.3f' % (epoch + 1, running_loss / len(dataloader), correct_predictions / total_samples))
        running_loss = 0.0
        # correct_predictions = 0
        # total_samples = 0

    print('Finished Training')

    # Save the model weights
    torch.save(model.state_dict(), weights_path)


# Create the dataset
valid_dataset = CustomDataset(csv_file=original_csv_path, img_dir=img_dir, transform=transform)

# Create a DataLoader for batching and shuffling (optional)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

model.eval()

# Initialize lists to store predictions and true labels
all_labels = []
all_outputs = []
all_predictions = []
all_names = []

# Iterate through the validation dataset
with torch.no_grad():
    for inputs, labels, names in tqdm(valid_dataloader):

        inputs, labels = inputs.to(device), labels.to(device)
        # import ipdb; ipdb.set_trace()
        # Forward pass to get predictions
        outputs = model(inputs)
        predicted_class = torch.argmax(outputs, dim=1)
        label_class = torch.argmax(labels, dim=1)

        # predictions = (outputs >= 0.5).float()  # Assuming a threshold of 0.5 for binary classification

        # Append predictions and true labels to the lists
        # all_outputs.extend(outputs.cpu().numpy()[:,0])
        all_predictions.extend(predicted_class.cpu().numpy()+1)
        all_labels.extend(label_class.cpu().float().numpy()+1)
        all_names.extend(names)
        # break

# Create a DataFrame with predictions and true labels
result_df = pd.DataFrame({'Image': all_names, 'Statistic': all_labels, 'Statistic_estimate': all_predictions})
# Save the DataFrame to a CSV file
result_df.to_csv(os.path.join(output_dir, 'amazon-name-true-statistic.csv'), index=False)

# use classification report to get the accuracy
print("Accuracy: {}".format(classification_report(all_labels, all_predictions)))


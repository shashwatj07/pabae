import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import CelebA
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

output_dir = '/scratch/aa117/data/celeba/results'

# Specify the path where you want to save the weights
weights_path = os.path.join(output_dir, 'mobilenetv2_smiling_classifier_blond.pth')
override = True

# Specify the path to the CelebA dataset
celeba_root = '/scratch/aa117/data/celeba/extracted_data/CelebA'
target_attribute = 'Smiling'
filter_attrs = ['Blond_Hair']

proxy_weights_path = os.path.join(output_dir, 'mobilenetv2_blond_classifier.pth')

'''
['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 
'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 
'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 
'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 
'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 
'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 
'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 
'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 
'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 
'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 
'Young', '']
'''

# Define a custom dataset for CelebA with binary labels for blond hair
class CelebABinaryDataset(CelebA):
    def __init__(self, root, attribute, filter_attrs=None, split='train', transform=None, target_transform=None, download=False):
        self.celeba_dataset = CelebA(root=root, split=split, target_type='attr', download=download)
        self.transform = transform
        self.target_transform = target_transform
        self.names = np.array(self.celeba_dataset.filename)
        # self.celeba_dataset = self.celeba_dataset
        # Extract 'Smiling' attribute for binary classification
        attr_idx = self.celeba_dataset.attr_names.index(attribute)
        self.labels = self.celeba_dataset.attr[:, attr_idx].float()
        if filter_attrs:
            for attr in filter_attrs:
                attr_idx = self.celeba_dataset.attr_names.index(attr)
                mask = self.celeba_dataset.attr[:, attr_idx] == 1
                self.celeba_dataset = torch.utils.data.Subset(self.celeba_dataset, torch.where(mask)[0])    
                self.labels = self.labels[mask]  
                self.names = self.names[mask.tolist()]      

    def __len__(self):
        return len(self.celeba_dataset)

    def __getitem__(self, idx):
        img, _ = self.celeba_dataset[idx]

        label = self.labels[idx]
        name = self.names[idx]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, label, name



# Define transformations for training (you can add more data augmentation if needed)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Create the custom dataset
celeba_dataset = CelebABinaryDataset(root=celeba_root, attribute=target_attribute, filter_attrs=filter_attrs, 
                                    transform=transform, split='train', download=True)
print(len(celeba_dataset))

# Create DataLoader for training
batch_size = 32
celeba_dataloader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

for inputs, targets, names in celeba_dataloader:
    print(inputs.size(), targets.size(), names)
    break

class FineTunedMobileNetV2(nn.Module):
    def __init__(self, num_classes=1):
        super(FineTunedMobileNetV2, self).__init__()

        # Load pre-trained MobileNetV2 model
        mobilenet_v2 = models.mobilenet_v2(weights='DEFAULT')
        
        # Extract the features (old layers) from MobileNetV2
        self.features = mobilenet_v2.features
        
        # Freeze the weights of the old layers
        for param in self.features.parameters():
            param.requires_grad = False

        # Define the new classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(1280, 128),  # Adjust input features based on the MobileNetV2 architecture
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        # Forward pass through the feature extractor
        x = self.features(x)
        
        # Forward pass through the classification head
        x = self.classification_head(x)
        
        return x

# Example usage:
# Create an instance of the model
model = FineTunedMobileNetV2(num_classes=1)
# proxy_moedl = FineTunedMobileNetV2(num_classes=1)

# Define the binary cross-entropy loss
criterion = nn.BCELoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1000, verbose=True)
# Training loop
num_epochs = 10

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))
model.to(device)
# proxy_model.to(device)

# import ipdb; ipdb.set_trace()
if (os.path.exists(weights_path) and not override) or num_epochs == 0:
    print("Loading weights from {}".format(weights_path))
    model.load_state_dict(torch.load(weights_path))
    print("Done")   
else:
    print("Training from scratch")
    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        # all_predictions = []
        # all_labels = []
        for i, data in enumerate(tqdm(celeba_dataloader)):
            inputs, labels, _ = data
            # print(inputs.shape)
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward, backward, and optimize
            outputs = model(inputs)
            # Assuming binary classification, adjust labels accordingly
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Update the learning rate scheduler based on the loss
            scheduler.step(loss)

            # Print statistics
            running_loss += loss.item()
            # import ipdb; ipdb.set_trace()
            # Calculate accuracy
            predictions = (outputs >= 0.5).float()  # Assuming a threshold of 0.5 for binary classification
            correct_predictions += torch.sum(predictions == labels).item()
            total_samples += labels.size(0)

            #print average loss and running accuracy every 10 mininbatches
            if (i+1) % 10 == 0:
                print('[Epoch: %d, %5d/ %d] Loss: %.3f  Accuracy: %.3f' % (epoch + 1, i + 1, len(celeba_dataloader), running_loss/(i+1), correct_predictions / total_samples))
            
            # if i % 100 == 99:  # Print every 100 mini-batches
        accuracy = correct_predictions / total_samples
        print('[Epoch: %d] Loss: %.3f  Accuracy: %.3f' % (epoch + 1, running_loss/len(celeba_dataloader), accuracy))
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        # all_predictions.extend(outputs.cpu().numpy())
        # all_labels.extend(labels.cpu().numpy())

    print('Finished Training')

    # Save the model weights
    torch.save(model.state_dict(), weights_path)


# Create the custom dataset
celeba_test = CelebABinaryDataset(root=celeba_root, attribute=target_attribute, filter_attrs=None,
                                    transform=transform, split='all', download=True)

# Create DataLoader for training
batch_size = 32
celeba_testloader = DataLoader(celeba_test, batch_size=batch_size, shuffle=True, num_workers=8)

model.eval()

# Initialize lists to store predictions and true labels
all_predictions = []
all_labels = []
all_outputs = []
all_names = []

# if os.path.exists(proxy_weights_path):
# print("Loading proxy weights from {}".format(proxy_weights_path))
# proxy_model.load_state_dict(torch.load(proxy_weights_path))
# proxy_model.eval()
# print("Done")  


# Iterate through the validation dataset
with torch.no_grad():
    for inputs, labels, names in tqdm(celeba_testloader):

        inputs, labels = inputs.to(device), labels.to(device)
        # Forward pass to get predictions
        outputs = model(inputs)
        # proxy_outputs = proxy_model(inputs)
        predictions = (outputs >= 0.5).float()  # Assuming a threshold of 0.5 for binary classification

        # Append predictions and true labels to the lists
        all_outputs.extend(outputs.cpu().numpy()[:,0])
        all_predictions.extend(predictions.cpu().numpy()[:,0])
        all_labels.extend(labels.cpu().numpy())
        all_names.extend(names)
        # break

# Create a DataFrame with predictions and true labels
result_df = pd.DataFrame({'Image': all_names, 'Statistic': all_labels, 'Statistic_estimate': all_predictions, 'Statistic_output': all_outputs})
# Save the DataFrame to a CSV file
result_df.to_csv(os.path.join(output_dir, 'celeba-all-statistic.csv'), index=False)

# import ipdb; ipdb.set_trace();
# Calculate accuracy
accuracy = accuracy_score(all_labels, all_predictions)

# Calculate precision
precision = precision_score(all_labels, all_predictions)

# Calculate recall
recall = recall_score(all_labels, all_predictions)

# Calculate F1 score
f1 = f1_score(all_labels, all_predictions)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
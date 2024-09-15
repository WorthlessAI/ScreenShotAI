import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler

from pynput.keyboard import Key

class ScreenshotDataset(Dataset):
    def __init__(self):
        self.data = torch.nn.functional.one_hot(torch.arange(0, len(Key)), len(Key)).to(dtype=torch.float32)
        print_screen_idx = [key.name for key in Key].index('print_screen')
        self.targets = torch.zeros(len(Key))
        self.targets[print_screen_idx] = 1.
        ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
        self.data_resampled, self.targets_resampled = ros.fit_resample(self.data, self.targets)
        self.data_resampled = torch.tensor(self.data_resampled, dtype=torch.float32)
        self.targets_resampled = torch.tensor(self.targets_resampled, dtype=torch.long)
        self.targets_resampled = torch.nn.functional.one_hot(self.targets_resampled, 2).to(dtype=torch.float32)

    def __len__(self):
        return len(self.data_resampled)

    def __getitem__(self, idx):
        x = self.data_resampled[idx]
        y = self.targets_resampled[idx]

        return x, y

class ScreenshotNN(nn.Module):
    def __init__(self):
        super(ScreenshotNN, self).__init__()
        self.fc = nn.Linear(in_features=len(Key), out_features=2)

    def forward(self, x):
        return self.fc(x)

# Training and validation functions
def train_model(model, dataloader, criterion, optimizer):
    model.train()  # Set the model to training mode
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        total_loss += loss.item() * inputs.size(0)  # Accumulate loss

        # Accuracy computation
        _, predicted = torch.max(outputs, 1)
        _, targets = torch.max(targets, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

def validate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in dataloader:
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            total_loss += loss.item() * inputs.size(0)  # Accumulate loss

            # Accuracy computation
            _, predicted = torch.max(outputs, 1)
            _, targets = torch.max(targets, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

# dataloader
dataset = ScreenshotDataset()
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# create model
model = ScreenshotNN()

class_weights = torch.tensor([1.0, 50.0], dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training and validation loop
num_epochs = 75
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_model(model, dataloader, criterion, optimizer)
    val_loss, val_accuracy = validate_model(model, dataloader, criterion)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

# save model
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('screenshot.pth') # Save

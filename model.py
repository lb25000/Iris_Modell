import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix 
import seaborn as sns
import matplotlib.pyplot as plt

# Sets all seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
# Define a custom Pytorch Dataset class
class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(), 
            nn.Linear(10,10), 
            nn.ReLU(), 
            nn.Linear(10, 3)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
# Method to train the model
def train(trainloader, model, loss_fn, optimizer, losses, fold):
    # Set model to train mode
    model.train()
    
    size = len(trainloader.dataset)
    
    for batch, (X, y) in enumerate(trainloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        losses[fold] = loss.item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print current loss after the model was trained with another ten batches
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Method to validate the model
def validate(validation_loader, model, results, fold):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
      for i, data in enumerate(validation_loader, 0):
        # Get inputs
        inputs, targets = data
        # Generate outputs
        outputs = model(inputs)
        # Set total and correct
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
      # Print accuracy
      print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
      print('--------------------------------')
      results[fold] = 100.0 * (correct / total)
      
# Method to test the model
def test(test_loader, model, loss_fn, epoch_testlosses, epoch_number):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
      for i, data in enumerate(test_loader, 0):
        # Get inputs
        inputs, targets = data
        # Generate outputs
        outputs = model(inputs)
        # Set total and correct
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
      epoch_testlosses[epoch_number] = loss_fn(outputs, targets)
      # Print accuracy
      print('Accuracy on test set: %d %%' % (100.0 * correct / total))
      print('--------------------------------')
    
if __name__ == '__main__':
    set_seed(26021997)
    
    # Load data and initialize features X and lables y
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Create IrisDataset object
    dataset = IrisDataset(X, y)
    
    # Define the split sizes (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    generator = torch.Generator().manual_seed(6)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    # Define the K-fold Cross Validator
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=25)
    
    # Define the loss function and optimizer
    # Use the cross entropy loss function because this is a multiclass classification problem
    loss_fn = nn.CrossEntropyLoss() 
    
    results = {}
    losses = {}
    model = NeuralNetwork()
        
    # Use the Adam otpimizer for optimization within back propagation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epoch_numbers = [25, 50, 75, 100, 125, 150, 175]
    epoch_testlosses = {}
    epoch_cvlosses = {}
    epoch_accs = {}
    for epoch_number in epoch_numbers:
    
        for fold, (train_ids, validation_ids) in enumerate(kfold.split(train_dataset)):
            print(f'Fold {fold}')
            print('---------------------------')
            
            # Sample elements randomly from a given list of ids, no replacement
            train_subsampler = SubsetRandomSampler(train_ids, generator=generator)
            validation_subsampler = SubsetRandomSampler(validation_ids, generator=generator)
            
            # Create DataLoader for batching and shuffling, batch_size 16 as the dataset consists of only 150 samples
            train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_subsampler)
            validation_loader = DataLoader(train_dataset, batch_size=16, sampler=validation_subsampler)
        
            # Training loop
            for t in range(epoch_number):
                print(f"Epoch {t+1}\n-------------------------------")
                train(train_loader, model, loss_fn, optimizer, losses, fold)
            
            print("Training process has finishid. Saving trained modell...")

            # Saving the model
            save_path = f'./model-fold-{fold}.pth'
            torch.save(model.state_dict(), save_path)
            
            # Test the model on the current validation set
            validate(validation_loader, model, results, fold)
        
        # Print results of training splits
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        sum_of_losses = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
        for key, value in losses.items():
            sum_of_losses += value
        epoch_accs[epoch_number] = sum/len(results.items())
        epoch_cvlosses[epoch_number] = sum_of_losses/len(losses.items())
        print(f'Average accuracy: {sum/len(results.items())} %')
        print(f'Cross validation error: {sum_of_losses/len(losses.items())}')

        # Test the model on test set and print result
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        test(test_loader, model, loss_fn, epoch_testlosses, epoch_number)
        
    # Visualize results
    # Get predictions on the test set for the confusion matrix
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in test_loader:
            # Forward pass: Get the predictions
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            
            # Append predictions and labels to the lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Create the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Convert dictionaries to lists for plotting
    test_losses = [epoch_testlosses[epoch].item() for epoch in epoch_numbers]
    cv_losses = [epoch_cvlosses[epoch] for epoch in epoch_numbers]
    accs = [epoch_accs[epoch] for epoch in epoch_numbers]
    
    # Set up the plots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the test losses
    sns.lineplot(x=epoch_numbers, y=test_losses, ax=axs[0], marker='o', label='Test Losses')
    sns.lineplot(x=epoch_numbers, y=cv_losses, ax=axs[0], marker='o', label='CV Losses')
    axs[0].set_title('Test and Cross-Validation Losses Over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot the accuracy
    sns.lineplot(x=epoch_numbers, y=accs, ax=axs[1], marker='o')
    axs[1].set_title('Average Accuracy of Folds Over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy (%)')

    # Display the plots
    plt.tight_layout()
    plt.show()

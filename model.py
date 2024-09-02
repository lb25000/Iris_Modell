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

def set_seed(seed):
    """
    Sets all seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class IrisDataset(Dataset):
    """
    Custom PyTorch Dataset class for the Iris dataset.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    """
    Simple neural network architecture with 4 input nodes,
    two hidden layers with 10 nodes each, and 3 output nodes,
    using ReLU activation functions.
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
    
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def train(train_loader, model, loss_fn, optimizer, losses, fold):
    """
    Trains the model in a training loop.
    
    Parameters:
    - train_loader: DataLoader for training
    - model: The model to be trained
    - loss_fn: Loss function
    - optimizer: Optimization algorithm
    - losses: Dictionary to store losses per fold
    - fold: Current fold index
    """
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        losses[fold] = loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print current loss every 10 batches
        if batch % 10 == 0:
            print(f"loss: {loss.item():>7f}  [{batch * len(X):>5d}/{len(train_loader.dataset):>5d}]")

def validate(validation_loader, model, results, fold):
    """
    Validates the model and saves results for the current fold.
    
    Parameters:
    - validation_loader: DataLoader for validation
    - model: The model to be validated
    - results: Dictionary to store results per fold
    - fold: Current fold index
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets in validation_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        accuracy = 100.0 * correct / total
        results[fold] = accuracy
        print(f'Accuracy for fold {fold}: {accuracy:.2f} %')

def test(test_loader, model, loss_fn, epoch_testlosses, epoch_number):
    """
    Tests the model and saves the loss for the given epoch number.
    
    Parameters:
    - test_loader: DataLoader for testing
    - model: The model to be tested
    - loss_fn: Loss function
    - epoch_testlosses: Dictionary to store test losses per epoch number
    - epoch_number: Current epoch number
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_testlosses[epoch_number] = loss_fn(outputs, targets)
        print(f'Accuracy on test set: {100.0 * correct / total:.2f} %')

if __name__ == '__main__':
    set_seed(26021997)

    # Load data and initialize features X and labels y
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
    
    model = NeuralNetwork()

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # Cross entropy for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with a small learning rate

    results = {}
    losses = {}

    epoch_numbers = range(10, 76, 5)
    epoch_testlosses = {}
    epoch_cvlosses = {}
    epoch_accs = {}

    for epoch_number in epoch_numbers:
        
        # Reset weights to train each model individually for the specified number of epochs
        model.apply(reset_weights)
        
        for fold, (train_ids, validation_ids) in enumerate(kfold.split(train_dataset)):
            print(f'Fold {fold}')
            print('---------------------------')

            # Create subset samplers for training and validation
            train_subsampler = SubsetRandomSampler(train_ids, generator=generator)
            validation_subsampler = SubsetRandomSampler(validation_ids, generator=generator)

            # Create DataLoader for batching and shuffling
            train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_subsampler)
            validation_loader = DataLoader(train_dataset, batch_size=16, sampler=validation_subsampler)

            # Train model for the specified number of epochs
            for t in range(epoch_number):
                print(f"Epoch {t+1}\n-------------------------------")
                train(train_loader, model, loss_fn, optimizer, losses, fold)

            print("Training process finished. Saving trained model...")

            # Save the model
            save_path = f'./model-fold-{fold}.pth'
            torch.save(model.state_dict(), save_path)

            # Validate the model on the current validation set
            validate(validation_loader, model, results, fold)

        # Print results of training splits
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        sum_accuracy = 0.0
        sum_of_losses = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value:.2f} %')
            sum_accuracy += value
        for key, value in losses.items():
            sum_of_losses += value

        epoch_accs[epoch_number] = sum_accuracy / len(results.items())
        epoch_cvlosses[epoch_number] = sum_of_losses / len(losses.items())
        print(f'Average accuracy: {epoch_accs[epoch_number]:.2f} %')
        print(f'Cross validation error: {epoch_cvlosses[epoch_number]:.4f}')

        # Test the model on the test set and print the result
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        test(test_loader, model, loss_fn, epoch_testlosses, epoch_number)

    # VISUALIZE RESULTS
    # Get predictions on the test set for the confusion matrix
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Create the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(10, 7))
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
    cv_loss_plot = sns.lineplot(x=epoch_numbers, y=cv_losses, ax=axs[0], marker='o', label='CV Losses')
    axs[0].set_title('Test and Cross-Validation Losses Over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    
    # Extract the color of the cv_losses line
    cv_loss_color = cv_loss_plot.lines[-1].get_color()

    # Plot the accuracy
    sns.lineplot(x=epoch_numbers, y=accs, ax=axs[1], marker='o', color=cv_loss_color)
    axs[1].set_title('Average Accuracy of Folds Over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy (%)')

    # Display the plots
    plt.tight_layout()
    plt.show()
    
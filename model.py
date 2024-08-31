import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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
            nn.Linear(4, 1),
            nn.ReLU(),
            nn.Linear(1, 3)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
# Method to train the model
def train(trainloader, model, loss_fn, optimizer):
    size = len(trainloader.dataset)
    model.train()
    
    for batch, (X, y) in enumerate(trainloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print current loss after the model was trained with another ten batches
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Method to test the model
def test(testloader, model, results, fold):
    correct, total = 0, 0
    
    with torch.no_grad():
        
      for i, data in enumerate(testloader, 0):
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
    
if __name__ == '__main__':
   
    # Load data and initialize features X and lables y
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Create IrisDataset object
    dataset = IrisDataset(X, y)
    
    model = NeuralNetwork()
    
    # Define the K-fold Cross Validator
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    epochs = 100
    
    # Define the loss function and optimizer
    # Use the cross entropy loss function because this is a multiclass classification problem
    loss_fn = nn.CrossEntropyLoss() 
    # TODO: Use the adam optimizer for gradient decent, why? 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    results = {}
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        
        print(f'Fold {fold}')
        print('---------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)
        
        # Create DataLoader for batching and shuffling, batch_size 16 as the dataset consists of only 150 samples
        train_loader = DataLoader(dataset, batch_size=16, sampler=train_subsampler)
        test_loader = DataLoader(dataset, batch_size=16, sampler=test_subsampler)
        
        # Training loop
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_loader, model, loss_fn, optimizer)
        
        print("Training process has finishid. Saving trained modell...")

        # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        torch.save(model.state_dict(), save_path)
                
        test(test_loader, model, loss_fn, fold)

    # Print results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')
        

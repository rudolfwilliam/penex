import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from typing import Tuple, Dict, List
import warnings

from penex.losses import PENEX

warnings.filterwarnings('ignore')

sensitivity = 0.3

class NeuralNetworkClassifier(nn.Module):
    """Simple linear classifier for multi-class classification."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class SimpleLinearClassifier(nn.Module):
    """Pure linear classifier (logistic regression)."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class PENEXLinearClassifier(SimpleLinearClassifier):
    """Linear classifier with PENEX loss."""
    def __init__(self, input_dim: int, num_classes: int, sensitivity: float = 0.1):
        super().__init__(input_dim, num_classes)
        self.sensitivity = sensitivity
    
    def forward(self, x):
        logits = super().forward(x)
        if not self.training:
            logits *= (1 + self.sensitivity)
        return logits

class PENEXNeuralNetworkClassifier(NeuralNetworkClassifier):
    """Linear classifier with PENEX loss."""
    def __init__(self, input_dim: int, num_classes: int, sensitivity: float = 0.1):
        super().__init__(input_dim, num_classes)
        self.sensitivity = sensitivity
    
    def forward(self, x):
        logits = super().forward(x)
        if not self.training:
            logits *= (1 + self.sensitivity)
        return logits
    

def load_classification_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load and preprocess classification datasets."""
    
    print(f"Loading {name} dataset...")
    
    if name == 'iris':
        X, y = load_iris(return_X_y=True)
        n_classes = 3
        
    elif name == 'wine':
        X, y = load_wine(return_X_y=True)
        n_classes = 3
        
    elif name == 'breast_cancer':
        X, y = load_breast_cancer(return_X_y=True)
        n_classes = 2
        
    elif name == 'sonar':
        try:
            sonar = fetch_openml('sonar', version=1, as_frame=True, parser='auto')
            X, y = sonar.data.values, sonar.target.values
            # Encode target labels
            le = LabelEncoder()
            y = le.fit_transform(y)
            n_classes = len(np.unique(y))
        except Exception as e:
            print(f"Error loading sonar dataset: {e}")
            print("Falling back to breast cancer dataset")
            return load_classification_dataset('breast_cancer')
            
    elif name == 'ionosphere':
        try:
            ionosphere = fetch_openml('ionosphere', version=1, as_frame=True, parser='auto')
            X, y = ionosphere.data.values, ionosphere.target.values
            le = LabelEncoder()
            y = le.fit_transform(y)
            n_classes = len(np.unique(y))
        except Exception as e:
            print(f"Error loading ionosphere dataset: {e}")
            print("Falling back to breast cancer dataset")
            return load_classification_dataset('breast_cancer')
            
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Handle any remaining categorical data
    if X.dtype == 'object':
        # Simple encoding for mixed data types
        le_encoder = LabelEncoder()
        for i in range(X.shape[1]):
            if X[:, i].dtype == 'object':
                X[:, i] = le_encoder.fit_transform(X[:, i])
        X = X.astype(float)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Dataset info:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {n_classes}")
    print(f"  Class distribution: {np.bincount(y_train)}")
    
    return X_train, X_test, y_train, y_test, n_classes


def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray, 
                       batch_size: int = 32, validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch data loaders with train/val/test splits."""
    
    # Split training data into train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=42, stratify=y_train
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_split)
    y_train_tensor = torch.LongTensor(y_train_split)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
             device: torch.device) -> Tuple[float, float]:
    """Validate model and return average loss and accuracy."""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def plot_comparison_curves(results_dict: Dict, dataset_name: str):
    """Plot comparison between CE and PENEX validation curves."""
    
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            pass  # Use default style if seaborn is not available
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = {'CE': 'blue', 'PENEX': 'red'}
    
    for model_name, model_results in results_dict.items():
        if model_results is None:
            continue
            
        training_history = model_results['training_history']
        val_losses = training_history['val_losses']
        val_accs = training_history['val_accs']
        epochs = range(1, len(val_losses) + 1)
        
        # Plot validation losses
        ax1.plot(epochs, val_losses, 
                color=colors.get(model_name, 'black'), 
                label=f'{model_name} Validation Loss', 
                linewidth=2, alpha=0.8)
        
        # Plot validation accuracies
        ax2.plot(epochs, val_accs, 
                color=colors.get(model_name, 'black'), 
                label=f'{model_name} Validation Accuracy', 
                linewidth=2, alpha=0.8)
    
    # Configure loss plot
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title(f'{dataset_name.title()} - Validation Loss Comparison', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Configure accuracy plot
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax2.set_title(f'{dataset_name.title()} - Validation Accuracy Comparison', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/{dataset_name}_ce_vs_penex_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                        train_accs: List[float], val_accs: List[float], 
                        dataset_name: str, model_type: str):
    """Plot individual training and validation curves (kept for single model analysis)."""
    
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            pass  # Use default style if seaborn is not available
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{dataset_name.title()} - {model_type} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{dataset_name.title()} - {model_type} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/{dataset_name}_{model_type.lower()}_training_curves.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                num_epochs: int, learning_rate: float, device: torch.device,
                dataset_name: str, model_type: str) -> Dict:
    """Complete training loop with validation."""
    
    train_criterion = nn.CrossEntropyLoss() if model_type == 'CE' else PENEX(sensitivity=sensitivity)
    test_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    
    # Track metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    best_model_state = None
    
    print(f"\nTraining {model_type} on {dataset_name}...")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, train_criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, test_criterion, device)
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1:3d}/{num_epochs}] | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Skip individual training curves - we'll plot comparisons later
    # plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
    #                     dataset_name, model_type)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, 
                  dataset_name: str, model_type: str):
    """Evaluate model on test set."""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    test_accuracy = accuracy_score(all_targets, all_predictions)
    
    print(f"\n=== {model_type} Results on {dataset_name.title()} ===")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions))
    
    return test_accuracy


def run_classification_experiments():
    """Run experiments on multiple classification datasets."""
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Configuration
    datasets = ['iris', 'wine', 'breast_cancer', 'sonar']
    num_epochs = 500
    learning_rate = 0.0001
    batch_size = 64
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            X_train, X_test, y_train, y_test, n_classes = load_classification_dataset(dataset_name)
            
            # Create data loaders
            train_loader, val_loader, test_loader = create_data_loaders(
                X_train, y_train, X_test, y_test, batch_size=batch_size
            )
            
            input_dim = X_train.shape[1]
            
            # Test both model types
            models = {
                'CE': NeuralNetworkClassifier(input_dim, n_classes),
                'PENEX': PENEXNeuralNetworkClassifier(input_dim, n_classes, sensitivity=sensitivity)
            }
            
            dataset_results = {}
            
            for model_name, model in models.items():
                model = model.to(device)
                
                # Train model
                training_history = train_model(
                    model, train_loader, val_loader, num_epochs, learning_rate, 
                    device, dataset_name, model_name
                )
                
                # Evaluate model
                test_accuracy = evaluate_model(model, test_loader, device, dataset_name, model_name)
                
                dataset_results[model_name] = {
                    'training_history': training_history,
                    'test_accuracy': test_accuracy
                }
            
            results[dataset_name] = dataset_results
            
            # Plot comparison curves for this dataset
            print(f"\nGenerating comparison plots for {dataset_name}...")
            plot_comparison_curves(dataset_results, dataset_name)
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name.upper()}:")
        for model_name, model_results in dataset_results.items():
            best_val_acc = model_results['training_history']['best_val_acc']
            test_acc = model_results['test_accuracy']
            print(f"  {model_name:6s}: Val Acc = {best_val_acc:.2f}%, Test Acc = {test_acc:.4f}")
    
    return results


if __name__ == "__main__":
    # Run all experiments
    results = run_classification_experiments()
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETED!")
    print("Check the 'plots' directory for training curve visualizations.")
    print("="*60)

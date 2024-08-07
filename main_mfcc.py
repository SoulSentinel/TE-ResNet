import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from model import create_model
from processor import AudioProcessor
from augment import AudioAugmenter
from evaluator import Evaluator

def update_json_results(new_results, filename='results.json'):
    # Check if file exists
    if os.path.exists(filename):
        # Read existing data
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Update data with new results
    data.update(new_results)

    # Write updated data back to file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"\nResults have been updated in {filename}")

class AudioDataset(Dataset):
    def __init__(self, processor, augmenter=None, augment_prob=0):
        self.processor = processor
        self.augmenter = augmenter
        self.augment_prob = augment_prob
        self.files = []
        self.labels = []

        genuine_dir = 'real'
        fake_dir = 'fake'

        for filename in os.listdir(genuine_dir):
            if filename.endswith('.wav'):
                self.files.append(os.path.join(genuine_dir, filename))
                self.labels.append(0)

        for filename in os.listdir(fake_dir):
            if filename.endswith('.wav'):
                self.files.append(os.path.join(fake_dir, filename))
                self.labels.append(1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]

        # Load and process audio
        audio, _ = self.processor.load_audio(audio_path)
        features = self.processor.extract_mfcc(audio)
        return torch.FloatTensor(features), label

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training MFCC"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(train_loader), accuracy

def evaluate(model, data_loader, criterion, device, evaluator):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating MFCC"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            
            # Get the predicted probabilities for the positive class
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            all_predictions.extend(probabilities)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(data_loader)
    
    # Use the Evaluator to compute EER
    evaluation_results = evaluator.evaluate(np.array(all_labels), np.array(all_predictions))
    eer = evaluation_results['EER']

    # Compute accuracy
    predictions = (np.array(all_predictions) > 0.5).astype(int)
    accuracy = (predictions == np.array(all_labels)).mean() * 100

    return avg_loss, accuracy, eer

def main():
    result = {}
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.0001
    num_epochs = 50
    test_size = 0.3# 20% for evaluation

    # Create instances of processor, augmenter, and evaluator
    processor = AudioProcessor()
    augmenter = AudioAugmenter()
    evaluator = Evaluator()

    # Create full dataset
    full_dataset = AudioDataset(processor, augmenter)

    # Split dataset into train and eval
    train_indices, eval_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=test_size, 
        stratify=full_dataset.labels,
        random_state=42
    )

    train_dataset = Subset(full_dataset, train_indices)
    eval_dataset = Subset(full_dataset, eval_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model
    model_config = {
        'input_channels': 1,
        'd_model': 72,
        'nhead': 8,
        'num_transformer_layers': 6,
        'num_classes': 2,
        'dropout': 0.1
    }
    model = create_model(**model_config).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/mfcc')

    # Training loop
    best_eval_eer = float('inf')
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        eval_loss, eval_acc, eval_eer = evaluate(model, eval_loader, criterion, device, evaluator)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.2f}%, Eval EER: {eval_eer:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Loss/Eval', eval_loss, epoch)
        writer.add_scalar('Accuracy/Eval', eval_acc, epoch)
        writer.add_scalar('EER/Eval', eval_eer, epoch)

        # Save the best model
        if eval_eer < best_eval_eer:
            best_eval_eer = eval_eer
            torch.save(model.state_dict(), 'best_model_mfcc.pth')

    # Close the TensorBoard writer
    writer.close()

    # Load the best model and perform final evaluation
    model.load_state_dict(torch.load('best_model_mfcc.pth'))
    final_eval_loss, final_eval_acc, final_eval_eer = evaluate(model, eval_loader, criterion, device, evaluator)
    print(f"Final Evaluation Loss: {final_eval_loss:.4f}, Eval Acc: {final_eval_acc:.2f}%, Eval EER: {final_eval_eer:.4f}")


    result["MFCC"] = {
        "loss": round(final_eval_loss, 4),
        "accuracy": round(final_eval_acc, 2),
        "eer": round(final_eval_eer, 4)
    }

    update_json_results(result)

if __name__ == "__main__":
    main()

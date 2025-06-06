import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class MinimalClassifier(nn.Module):
    def __init__(self, input_len=512, num_classes=7):
        super(MinimalClassifier, self).__init__()
        self.fc = nn.Linear(input_len, num_classes)
    def forward(self, x):
        # x: (batch, 1, seq_len)
        x = x.squeeze(1)  # (batch, seq_len)
        return self.fc(x)

def load_dataset(dataset_path):
    data = np.load(dataset_path)
    x_noisy = torch.FloatTensor(data['x_noisy'])
    if x_noisy.ndim == 2:
        x_noisy = x_noisy.unsqueeze(1)
    labels = torch.LongTensor(data['labels'])
    return x_noisy, labels

def main():
    DATASET_PATH = "datasets/cnn_dataset.npz"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 20
    print("Minimal classifier sanity check!")
    x_noisy, labels = load_dataset(DATASET_PATH)
    # Для быстрой проверки возьмём только 1000 примеров
    x_noisy = x_noisy[:1000]
    labels = labels[:1000]
    x_train, x_val, y_train, y_val = train_test_split(x_noisy, labels, test_size=0.2, stratify=labels, random_state=42)
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = MinimalClassifier(input_len=x_noisy.shape[2], num_classes=7).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} (train)', leave=False):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} (val)', leave=False):
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

if __name__ == "__main__":
    main() 
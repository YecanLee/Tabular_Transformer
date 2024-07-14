import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

import wandb
from tabformer.tabtranformer import TabTransformer

# Load the training data
train = pd.read_csv('data/processed/train_new.csv')


# Handle missing values
for i in train.columns:
    if train[i].dtype == 'object':
        train[i].fillna(train[i].mode()[0], inplace=True)
    else:
        train[i].fillna(train[i].mean(), inplace=True)

# Encoding categorical variables
label_encoders = {}
categorical_columns = train.select_dtypes(include=['object']).columns.drop('ID')
for i in categorical_columns:
    le = LabelEncoder()
    train[i] = le.fit_transform(train[i])
    label_encoders[i] = le


# Normalizing numerical features
scaler = MinMaxScaler()
numeric_columns = train.select_dtypes(include=[np.number]).columns.drop('Yield')
train[numeric_columns] = scaler.fit_transform(train[numeric_columns])

# Calculate mean and standard deviation of continuous features
continuous_mean = train[numeric_columns].mean().values
continuous_std = train[numeric_columns].std().values

# Preparing the data for modeling
X_categ = train[categorical_columns]
X_cont = train[numeric_columns]
y = train['Yield']

# Splitting the dataset into training and validation sets
X_categ_train, X_categ_test, X_cont_train, X_cont_test, y_train, y_test = train_test_split(
    X_categ, X_cont, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_categ_train_tensor = torch.tensor(X_categ_train.values, dtype=torch.long)
X_categ_test_tensor = torch.tensor(X_categ_test.values, dtype=torch.long)
X_cont_train_tensor = torch.tensor(X_cont_train.values, dtype=torch.float32)
X_cont_test_tensor = torch.tensor(X_cont_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Create DataLoaders
train_dataset = TensorDataset(X_categ_train_tensor, X_cont_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_categ_test_tensor, X_cont_test_tensor, y_test_tensor)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize wandb
wandb.login()
wandb.init(
    project='zindi-crop-challenge',
    name='tabular-transformer-3',
    config={
        'learning_rate': 1e-4,
        'batch_size': batch_size,
        'num_epochs': 300,
        'model_architecture': 'TabTransformer',
        'embedding_dim': 32,
        'num_heads': 4,
        'num_layers': 2,
        'dropout': 0.1
    },
    tags=['baseline', 'tabular-data'],
    notes='Initial experiment with TabTransformer on Zindi Crop Challenge dataset'
)

# Define the TabTransformer model
input_dim = X_cont.shape[1]
output_dim = 1
embedding_dim = 32
num_heads = 4
num_layers = 2
dropout = 0.1

model = TabTransformer(
    categories=[len(le.classes_) for le in label_encoders.values()],
    num_continuous=input_dim,
    dim=embedding_dim,
    depth=num_layers,
    heads=num_heads,
    attn_dropout=dropout,
    ff_dropout=dropout,
    dim_out=output_dim,
    mlp_act=nn.ReLU(),
    continuous_mean_std=torch.tensor(np.stack([continuous_mean, continuous_std], axis=1), dtype=torch.float32),
)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Train the model
num_epochs = 300
patience = 10
best_val_loss = float('inf')
counter = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(num_epochs):
    # Training loop
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        batch_categ, batch_cont, batch_y = batch
        batch_categ = batch_categ.to(device)
        batch_cont = batch_cont.to(device).float()
        batch_y = batch_y.to(device).float()

        # Forward pass
        outputs = model(batch_categ, batch_cont)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_categ.size(0)

    train_loss /= len(train_dataloader.dataset)
    train_rmse = math.sqrt(train_loss)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            batch_categ, batch_cont, batch_y = batch
            batch_categ = batch_categ.to(device)
            batch_cont = batch_cont.to(device).float()
            batch_y = batch_y.to(device).float()

            # Forward pass
            outputs = model(batch_categ, batch_cont)
            loss = criterion(outputs, batch_y)

            val_loss += loss.item() * batch_categ.size(0)

    val_loss /= len(test_dataloader.dataset)
    val_rmse = math.sqrt(val_loss)

    # Log metrics to wandb
    wandb.log({'train_loss': train_rmse, 'val_loss': val_rmse, 'epoch': epoch})

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# Save the final model
torch.save(model.state_dict(), 'final_model.pth')
wandb.finish()
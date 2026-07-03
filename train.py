import torch
import torch.nn as nn

from models.House_Price_Pred import HousePriceModel
from utils.preprocess import load_data

from utils.dataset import HouseDataset

from torch.utils.data import DataLoader




X_train, X_test, y_train, y_test, scaler = load_data()

train_dataset = HouseDataset(
    X_train,
    y_train
)

print(len(train_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

model = HousePriceModel()

model.load_state_dict(torch.load("saved_model.pth"))  #call the saved model to continue training

print("previous model loaded successfully")

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 100

for epoch in range(epochs):

    model.train()

    for X_batch, y_batch in train_loader:

        prediction = model(X_batch)

        loss = loss_fn(prediction, y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1} Loss = {loss.item():.4f}")


test_dataset = HouseDataset(X_test, y_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

model.eval()

test_loss = 0

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        prediction = model(X_batch)

        loss = loss_fn(prediction, y_batch)

        test_loss += loss.item()

test_loss /= len(test_loader)

print(f"Test Loss: {test_loss:.4f}")      #average loss


torch.save(
    model.state_dict(),
    "saved_model.pth"
)

print("Model Saved")
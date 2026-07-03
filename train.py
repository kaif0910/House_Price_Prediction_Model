import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models.House_Price_Pred import HousePriceModel
from utils.preprocess import load_data

from utils.dataset import HouseDataset

from torch.utils.data import DataLoader

import joblib

from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score
)




X_train, X_test, X_val, y_train, y_test, y_val, scaler = load_data()
joblib.dump(scaler,"scaler.pkl")


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

val_dataset = HouseDataset(
    X_val,
    y_val
)


val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
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

train_losses = []
val_losses = []

for epoch in range(epochs):

    model.train()
    train_loss = 0 

    for X_batch, y_batch in train_loader:

        prediction = model(X_batch)

        loss = loss_fn(prediction, y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)

    #validation

    model.eval()
    val_loss = 0 

    with torch.no_grad():

        for X_batch, y_batch in val_loader:
            prediction = model(X_batch)
            loss = loss_fn(prediction, y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    #save losses

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Validation Loss : {val_loss:.4f}")
    
    print("-" * 40)


plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Learning Curve")
plt.legend()
plt.show()




test_dataset = HouseDataset(X_test, y_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

model.eval()

test_loss = 0

all_predictions = []
all_actuals = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        prediction = model(X_batch)

        all_predictions.extend(
            prediction.numpy()
        )

        all_actuals.extend(
            y_batch.numpy()
        )


        loss = loss_fn(prediction, y_batch)

        test_loss += loss.item()

test_loss /= len(test_loader)

print(f"Test Loss: {test_loss:.4f}")      #average loss

        
mae = mean_absolute_error(all_actuals, all_predictions)
mse = mean_squared_error(all_actuals, all_predictions)
rmse = mse ** 0.5
r2 = r2_score(all_actuals, all_predictions)


print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2 : {r2:.4f}")


torch.save(
    model.state_dict(),
    "saved_model.pth"
)

print("Model Saved")
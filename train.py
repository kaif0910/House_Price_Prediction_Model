import torch
import torch.nn as nn

from models.House_Price_Pred import HousePriceModel
from utils.preprocess import load_data


X_train, X_test, y_train, y_test, scaler = load_data()

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

    prediction = model(X_train)

    loss = loss_fn(
        prediction,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1} Loss = {loss.item():.4f}"
        )

model.eval()

with torch.no_grad():

    prediction = model(X_test)

    test_loss = loss_fn(
        prediction,
        y_test
    )

print("Test Loss:", test_loss.item())

torch.save(
    model.state_dict(),
    "saved_model.pth"
)

print("Model Saved")
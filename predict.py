import torch

from models.House_Price_Pred import HousePriceModel
from utils.preprocess import load_data


_, X_test, _, y_test, _= load_data()

model = HousePriceModel()

model.load_state_dict(
    torch.load("saved_model.pth")
)

model.eval()


with torch.no_grad():

    prediction = model(X_test)

    for i in range(10):
        print(
            f"House{i+1}"
        )
        print(f"predicted: {prediction[i].item():.2f}")
        print(f"actual: {y_test[i].item():.2f}")
        print("-" * 30)


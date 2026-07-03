import torch

from models.House_Price_Pred import HousePriceModel
from utils.preprocess import load_data

import joblib
scaler = joblib.load("scaler.pkl")


# _, X_test, _, y_test, _= load_data()

med_inc = float(input("Median Income: "))
house_age = float(input("House Age: "))
ave_rooms = float(input("Average Rooms: "))
ave_bedrooms = float(input("Average Bedrooms: "))
population = float(input("Population: "))
ave_occup = float(input("Average Occupancy: "))
latitude = float(input("Latitude: "))
longitude = float(input("Longitude: "))

house = [[
    med_inc,
    house_age,
    ave_rooms,
    ave_bedrooms,
    population,
    ave_occup,
    latitude,
    longitude
]]

house = scaler.transform(house)

house = torch.tensor(
    house,
    dtype=torch.float32
)

model = HousePriceModel()

model.load_state_dict(
    torch.load("saved_model.pth")
)

model.eval()


with torch.no_grad():

    prediction = model(house)

    print("predicted Price: ", prediction.item())

    # for i in range(10):
    #     print(
    #         f"House{i+1}"
    #     )
    #     print(f"predicted: {prediction[i].item():.2f}")
    #     print(f"actual: {y_test[i].item():.2f}")
    #     print("-" * 30)


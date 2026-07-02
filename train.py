import torch 
import torch.nn as nn
import pandas as pd   #data Frames

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["price"] = housing.target

X = df.drop("price", axis=1)  #input only 8 columns

y = df["price"]   #output = label

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(
    y_train.values,
    dtype= torch.float32
).unsqueeze(1)

y_test = torch.tensor(
    y_test.values,
    dtype= torch.float32
).unsqueeze(1)

class HousePriceModel(nn.module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(8,16),

            nn.ReLU(),

            nn.Linear(16,1)

        )

    def forward(self,x):
        return self.network(x)

model = HousePriceModel()


loss_fn = nn.MSELoss()  #loss function

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 100

for epoch in range(epochs):
    model.train()
    prediction = model(X_train)
    loss = loss_fn(prediction,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(epoch + 1) % 10 == 0 :
        print(
            f"Epoch {epoch + 1}, Loss: {loss.item(): .4f}"
        )
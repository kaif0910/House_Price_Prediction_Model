from torch.nn import nn

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
from torch import nn
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class NominalModel(nn.Module):
    nominal_y0 = torch.tensor([[2., 0.]]).to(device)
    nominal_A = torch.tensor([[-0.1, 2.5], [-2.5, -0.1]]).to(device)

    def forward(self, t, y):
        if y.dim() == 2:
            # return torch.mm(y ** 3, NominalModel.nominal_A)
            return torch.mm(y ** 1, NominalModel.nominal_A)

        else:
            # print(y.shape)
            batch_A = NominalModel.nominal_A.unsqueeze(0).repeat(y.shape[0], 1, 1)
            # print(batch_A.shape)
            # return torch.bmm(y ** 3, batch_A)
            return torch.bmm(y ** 1, batch_A)



class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):

        model = NominalModel().forward(t, y)
        nn_model = self.net(y ** 3)

        return model + nn_model

import torch

class MNISTNet(torch.nn.Module):
  def __init__(self):
    super(MNISTNet, self).__init__()
    self.fc1 = torch.nn.Linear(28*28, 250)
    self.act1 = torch.nn.Sigmoid()
    self.fc2 = torch.nn.Linear(250,250)
    self.act2 = torch.nn.Sigmoid()
    self.fc3 = torch.nn.Linear(250, 10)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act1(x)
    x = self.fc2(x)
    x = self.act2(x)
    x = self.fc3(x)
    return x

"""
Implementation from https://github.com/rgeirhos/lucent/blob/dev/lucent/modelzoo/inceptionv1/InceptionV3.py

Main use to discriminator between real and visualization images in order to implment the fooling circuit
"""
class SimpleCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(3, 16, 3),
      nn.ReLU(),
      nn.Conv2d(16, 16, 5, 2),
      nn.ReLU(),
      nn.Conv2d(16, 16, 5, 2),
      nn.ReLU(),
      nn.Conv2d(16, 16, 5, 2),
      nn.ReLU(),
      nn.Conv2d(16, 16, 5, 2),
      nn.ReLU(),
      nn.Conv2d(16, 3, 3, 1),
      nn.ReLU(),
      Flatten(),
      nn.Dropout(0.3),
      nn.Linear(243, 1),
    )

  def forward(self, x):
    return self.layers(x)
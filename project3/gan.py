import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# hyperparameters
BATCH_SIZE = 128
Z_DIM = 100
OUTPUT_DIM = 28 * 28
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
# normalize the data to [-1, 1] range to match Generator's tanh output
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

train_loader = DataLoader(mnist_trainset, BATCH_SIZE, shuffle=True, drop_last=True)

# Create a Generator class.
class Generator(nn.Module):
    def __init__(self, Z_DIM, OUTPUT_DIM):
        super(Generator, self).__init__()
        # Define your network architecture.
        self.net = nn.Sequential(
            # Layer 1: z_dim -> 256
            nn.Linear(Z_DIM, 256),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 256 -> 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 512 -> 1024
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer: 1024 -> 784 (28*28), squash to [-1, 1]
            nn.Linear(1024, OUTPUT_DIM),
            nn.Tanh()
        )

    def forward(self, x):
        # Define your network data flow. 
        return self.net(x)
    

# Create a Generator.
netG = Generator(Z_DIM, OUTPUT_DIM).to(DEVICE)

# Create a Discriminator class.
class Discriminator(nn.Module):
    def __init__(self, input_dim=OUTPUT_DIM):
        super(Discriminator, self).__init__()
        # Define your network architecture.
        self.net = nn.Sequential(
            # Layer 1: 784 -> 512
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Layer 2: 512 -> 256
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Output layer: 256 -> 1 (probability of being real)
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Define your network data flow. 
        return self.net(x)
    
# Create a Discriminator.
netD = Discriminator().to(DEVICE)

# Setup binary cross-entropy loss function
criterion = nn.BCELoss()

# Setup Generator optimizer.
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Setup Discriminator optimizer.
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define loss function. 
criterion = torch.nn.BCELoss()
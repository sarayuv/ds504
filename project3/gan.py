import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# hyperparameters
BATCH_SIZE = 128

# load data
# normalize the data to [-1, 1] range to match Generator's tanh output
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

train_loader = DataLoader(mnist_trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Create a Generator class.
class Generator(nn.Module):
    def __init__(self, ):
        super(Generator, self).__init__()
        # Define your network architecture.

    def forward(self, x):
        # Define your network data flow. 
        return output
# Create a Generator.
netG = Generator(*args)

# Create a Discriminator class.
class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        # Define your network architecture.

    def forward(self, x):
        # Define your network data flow. 
        return output
# Create a Discriminator.
netD = Discriminator(*args)

# Setup Generator optimizer.
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.9, 0.999))

# Setup Discriminator optimizer.
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.9, 0.999))

# Define loss function. 
criterion = torch.nn.BCELoss()

# Training
def train():
    for _ in range(batchCount):  
	
        # Create a batch by drawing random index numbers from the training set
       
        # Create noise vectors for the generator
        
        # Generate the images from the noise

        # Create labels

        # Train discriminator on generated images

        # Train generator

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# store all hyperparameters and configuration for the GAN training/architecture
@dataclass
class GANConfig:
    # define data dimensions
    batch_size: int = 128
    z_dim: int = 100
    img_channels: int = 1
    img_size: int = 28

    # define architecture of the generator and discriminator
    g_proj_channels: int = 128
    g_mid_channels: int = 64
    d_channels1: int = 64
    d_channels2: int = 128
    d_dropout: float = 0.2
    g_activation: str = "relu"
    d_activation: str = "leaky_relu"

    # define learning rates and optimizer parameters
    g_lr: float = 2e-4
    d_lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    optimizer: str = "adam"

    # define loss function and weight initialization scheme
    loss: str = "bce_logits"
    weight_init: str = "dcgan"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# converts dataclass to dictionary for easier logging/debugging
def config_to_dict(config: GANConfig) -> dict:
    return asdict(config)

# builds the dataloader for the training set with transformations
def build_train_loader(batch_size: int, data_root: str = "./data") -> DataLoader:
    transform = transforms.Compose(
        [
            # convert images to tensors
            transforms.ToTensor(),
            # normalize pixel values to [-1, 1]
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

# returns the activation function based on the name specified in the config
def _activation(name: str, negative_slope: float = 0.2) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")

# defines the generator and discriminator architectures based on the config parameters
class Generator(nn.Module):
    def __init__(self, config: GANConfig):
        super().__init__()
        self.config = config
        c0 = config.g_proj_channels
        c1 = config.g_mid_channels

        # fully connected layer to project the noise vector to a feature map and reshape it for the convolutional layers
        self.proj = nn.Sequential(
            nn.Linear(config.z_dim, c0 * 7 * 7),
            nn.BatchNorm1d(c0 * 7 * 7),
            _activation(config.g_activation),
        )

        # build the convolutional layers to upsample the feature map to the image size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c0, c1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            _activation(config.g_activation),
            nn.ConvTranspose2d(c1, config.img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    # project the noise vector into a feature map and upsample it to generate an image
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z)
        x = x.view(-1, self.config.g_proj_channels, 7, 7)
        return self.net(x)


# defines the discriminator architecture based on the config parameters
class Discriminator(nn.Module):
    def __init__(self, config: GANConfig):
        super().__init__()
        c1 = config.d_channels1
        c2 = config.d_channels2

        # build the convolutional layers to downsample the input image and extract features for classification
        self.features = nn.Sequential(
            nn.Conv2d(config.img_channels, c1, kernel_size=4, stride=2, padding=1),
            _activation(config.d_activation),
            nn.Dropout2d(config.d_dropout),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            _activation(config.d_activation),
            nn.Dropout2d(config.d_dropout),
        )
        self.classifier = nn.Linear(c2 * 7 * 7, 1)

    # flatten the feature map and pass it through the classifier to get the output logit for real/fake classification
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# initializes the weights of the model based on the schema
def initialize_weights(model: nn.Module, schema: str = "dcgan") -> None:
    schema = schema.lower()

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            # normal initialization
            if schema == "dcgan":
                nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
            # xavier initialization for better convergence
            elif schema == "xavier":
                nn.init.xavier_normal_(module.weight.data)
            # kaiming initialization for relu-based networks
            elif schema == "kaiming":
                nn.init.kaiming_normal_(module.weight.data, nonlinearity="leaky_relu")
            else:
                raise ValueError(f"Unsupported weight init schema: {schema}")

            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0.0)

        # batch norm layers should be initialized with mean=1 and std=0 for the weights, and 0 for the biases
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(module.bias.data, 0.0)


# returns the loss function specified by loss_name, supports binary cross-entropy with logits
def build_criterion(loss_name: str) -> nn.Module:
    name = loss_name.lower()
    if name == "bce_logits":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unsupported loss function: {loss_name}")


# creates the optimizer based on the config parameters
def _build_optimizer(name: str, params, lr: float, beta1: float, beta2: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


# constructs the generator, discriminator, loss function, and optimizers
def build_gan_components(config: GANConfig, device: torch.device = DEVICE):
    # create generator and disciminator instances
    net_g = Generator(config).to(device)
    net_d = Discriminator(config).to(device)

    # initilize the weights of the models using the specified schema
    initialize_weights(net_g, config.weight_init)
    initialize_weights(net_d, config.weight_init)

    # build the loss function based on the config
    criterion = build_criterion(config.loss)

    # create optimizers for the generator and discriminator with the specified learning rates and parameters
    optimizer_g = _build_optimizer(
        config.optimizer,
        net_g.parameters(),
        config.g_lr,
        config.beta1,
        config.beta2,
    )
    optimizer_d = _build_optimizer(
        config.optimizer,
        net_d.parameters(),
        config.d_lr,
        config.beta1,
        config.beta2,
    )

    return net_g, net_d, criterion, optimizer_g, optimizer_d

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


# A function for weight Initialization
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)
        torch.nn.init.constant_(m.bias.data, 0)

# FR-GAN Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
 
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
#             *block(3, 32, normalize=False),
#             nn.Linear(32, 1),
            nn.Linear(3,1)
        )

    def forward(self, z):
        output = self.model(z)
        return output

    
# FR-GAN Dicriminator for Fairness
class Discriminator_F(nn.Module):
    def __init__(self):
        super(Discriminator_F, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1,1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity
    
    
# FR-GAN Dicriminator for Robustness
class Discriminator_R(nn.Module):
    def __init__(self):
        super(Discriminator_R, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(4, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity
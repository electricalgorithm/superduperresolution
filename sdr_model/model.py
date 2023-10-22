"""
This module contains the model class for the project.
"""
import logging

from torch import Tensor
from torch import nn

# Create a logger.
logger = logging.getLogger(__name__)

class SuperDuperResolution(nn.Module):
    """
    SRCNN model taken from github@kawaiibilli/pytorch_srcnn
    """
    def __init__(self):
        super(SuperDuperResolution, self).__init__()
        # Layers from SRCNN paper.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x) -> Tensor:
        """Forward pass function for calculating the output.
        Args:
            x: input image
        Returns:
            out: output image
        """
        out = self.conv1(x)
        logger.debug("First convolutional layer is applied.")
        logger.debug("Shape of the output: %s", str(out.shape))

        out = self.relu1(out)
        logger.debug("ReLU activation function is applied.")
        logger.debug("Shape of the output: %s", str(out.shape))

        out = self.conv2(out)
        logger.debug("Second convolutional layer is applied.")
        logger.debug("Shape of the output: %s", str(out.shape))

        out = self.relu2(out)
        logger.debug("ReLU activation function is applied.")
        logger.debug("Shape of the output: %s", str(out.shape))

        out = self.conv3(out)
        logger.debug("Third convolutional layer is applied.")
        logger.debug("Shape of the output: %s", str(out.shape))

        return out

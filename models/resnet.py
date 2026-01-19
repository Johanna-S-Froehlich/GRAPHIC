import timm
import os
import torch.nn as nn
from torch import Tensor
from huggingface_hub import hf_hub_download
from typing import Any, Optional, Callable, List, Type
import torch

# ResNet implementation from torchvision models slightly modified to fit the CIFAR-100 dataset 
# Source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """
    Creates a 3x3 convolutional layer with padding.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride value for the convolution (default: 1).
                                Controls how much the filter moves across the input.
        groups (int, optional): Number of groups for grouped convolution (default: 1).
                                Groups separate the input and output channels into 'groups' for increased efficiency.
                                When groups=1, it behaves as a standard convolution.
        dilation (int, optional): Spacing between kernel elements (default: 1).
                                  Higher dilation increases the receptive field without increasing the kernel size.

    Returns:
        nn.Conv2d: A 2D convolutional layer with specified parameters.

    Notes:
        - Kernel size: 3x3 (standard for capturing spatial features while maintaining small receptive fields).
        - Padding: Automatically set to match dilation, ensuring the output size is controlled.
        - Bias: Disabled (bias=False) because the subsequent Batch Normalization layer handles bias.
        - Used in the ResNet architecture for feature extraction.
    """
    return nn.Conv2d(
        in_planes,               # Input channel dimension.
        out_planes,              # Output channel dimension.
        kernel_size=3,           # 3x3 convolution kernel.
        stride=stride,           # Stride for downsampling or keeping spatial dimensions.
        padding=dilation,        # Padding to maintain spatial dimensions (adjusted for dilation).
        groups=groups,           # Supports grouped convolutions (e.g., ResNeXt-style networks).
        bias=False,              # No bias since BatchNorm handles it.
        dilation=dilation,       # Controls spacing between kernel elements.
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    Creates a 1x1 convolutional layer.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride value for the convolution (default: 1).
                                Controls downsampling if stride > 1.

    Returns:
        nn.Conv2d: A 1x1 convolutional layer with specified parameters.

    Notes:
        - Kernel size: 1x1 (point-wise convolution).
        - No padding needed as the kernel is 1x1.
        - Bias: Disabled (bias=False) as Batch Normalization is applied after this layer.
        - 1x1 convolutions are used in ResNet for:
            - Reducing or increasing channel dimensions (bottleneck structure).
            - Efficient feature transformation.
            - Handling residual connections when dimensions change.
    """
    return nn.Conv2d(
        in_planes,          # Input channel dimension.
        out_planes,         # Output channel dimension.
        kernel_size=1,      # 1x1 convolution kernel (point-wise convolution).
        stride=stride,      # Controls downsampling.
        bias=False          # Bias is not needed due to Batch Normalization.
    )

class Bottleneck(nn.Module):
    # The Bottleneck block is a key building block for ResNet models.
    # This implementation follows the ResNet V1.5 variant (also known as "Bottleneck" in torchvision)
    # where the stride for downsampling is placed at the 3x3 convolution (conv2) instead of the first 1x1 (conv1).
    # This design choice was motivated by improved accuracy (see https://arxiv.org/abs/1512.03385 and NVIDIA's implementation).

    expansion: int = 4  # Factor by which the number of channels is increased at the block's output.

    def __init__(
            self,
            inplanes: int,                           # Number of input channels.
            planes: int,                             # Number of output channels (before expansion).
            stride: int = 1,                         # Stride to apply in the 3x3 convolution; controls downsampling.
            downsample: Optional[nn.Module] = None,  # Optional downsampling layer to match dimensions for the skip connection.
            groups: int = 1,                         # Number of groups for grouped convolution (default is 1: standard convolution).
            base_width: int = 64,                      # Base width used to calculate the channel width.
            dilation: int = 1,                       # Dilation for the 3x3 convolution (affects receptive field).
            norm_layer: Optional[Callable[..., nn.Module]] = None,  # Normalization layer to use (default is BatchNorm2d).
    ) -> None:
        super().__init__()
        # If no normalization layer is provided, default to BatchNorm2d.
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Calculate the width (number of channels) for the block's internal convolutions.
        # This width is scaled by base_width and the number of groups.
        width = int(planes * (base_width / 64.0)) * groups

        # The first convolution: 1x1 convolution to reduce dimensionality.
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        # The second convolution: 3x3 convolution.
        # This is where downsampling is performed if stride != 1.
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        # The third convolution: 1x1 convolution to expand the number of channels.
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        # LeakyReLU activation function is used for non-linearity.
        self.relu = nn.LeakyReLU(inplace=True)

        # Downsample layer to match the identity's shape to the output shape if needed.
        self.downsample = downsample

        # Save the stride value for potential use elsewhere.
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        # Save the original input for the skip connection.
        identity = x

        # First convolution block: 1x1 convolution -> BatchNorm -> ReLU activation.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolution block: 3x3 convolution (with stride) -> BatchNorm -> ReLU activation.
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Third convolution block: 1x1 convolution -> BatchNorm.
        out = self.conv3(out)
        out = self.bn3(out)

        # If a downsample layer is provided, apply it to the original input to match dimensions.
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the skip connection (the identity) to the output.
        out += identity

        # Apply the final activation function.
        out = self.relu(out)

        # Return the final output of the Bottleneck block.
        return out

class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Bottleneck],  # Residual block type (e.g., Bottleneck block).
            layers: List[int],        # Number of blocks in each layer (e.g., [3, 4, 6, 3] for ResNet-50).
            num_classes: int = 1000,  # Number of output classes (default: 1000 for ImageNet).
            zero_init_residual: bool = False,  # If True, initializes the last BN layer of each block to 0.
            groups: int = 1,          # Number of groups for grouped convolutions (for ResNeXt models).
            width_per_group: int = 64, # Base width of each group (for ResNeXt models).
            replace_stride_with_dilation: Optional[List[bool]] = None, # Replace stride with dilation (for dilation in the last layers).
            norm_layer: Optional[Callable[..., nn.Module]] = None,  # Normalization layer (default: BatchNorm2d).
    ) -> None:
        super().__init__()

        # Set default normalization layer if not provided (BatchNorm2d).
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # Initial input channel size.
        self.inplanes = 64  
        self.dilation = 1  # Dilation rate for convolutions.

        # Handle the stride-dilation configuration for the last three layers.
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        # Parameters for grouped convolution (for ResNeXt models).
        self.groups = groups
        self.base_width = width_per_group

        # Initial convolution: 7x7 kernel with stride 2 (downsampling the image size by 2).
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)  # Batch Normalization after the first convolution.
        self.relu = nn.LeakyReLU(inplace=True)  # LeakyReLU activation (in-place to save memory).

        # Max pooling to further reduce spatial dimensions by 2.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create the ResNet layers using the _make_layer method.
        self.layer1 = self._make_layer(block, 64, layers[0])  # First residual layer (no downsampling).
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])  
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])  
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])  

        # Global Average Pooling to reduce feature map to 1x1 spatial dimensions.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected (FC) layer for classification output.
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize convolution weights using Kaiming Normal (He initialization).
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # Initialize normalization layers with weight=1 and bias=0.
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Optionally zero-initialize the last batch norm layer in each Bottleneck block for stability.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # Zero-initialize the last BN layer.

    def _make_layer(
            self,
            block: Type[Bottleneck],  # Residual block type (e.g., Bottleneck).
            planes: int,              # Number of output channels for this layer.
            blocks: int,              # Number of blocks in the layer.
            stride: int = 1,          # Stride for downsampling.
            dilate: bool = False,     # Whether to use dilation instead of stride.
    ) -> nn.Sequential:
        """
        Constructs one layer of the ResNet model with multiple residual blocks.

        Args:
            block (Type[Bottleneck]): The type of residual block (Bottleneck).
            planes (int): Number of output channels.
            blocks (int): Number of blocks in this layer.
            stride (int, optional): Stride for the first block. Defaults to 1.
            dilate (bool, optional): Whether to apply dilation instead of stride. Defaults to False.

        Returns:
            nn.Sequential: A sequential container of residual blocks.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        # Apply dilation instead of stride if specified.
        if dilate:
            self.dilation *= stride
            stride = 1

        # Create downsampling layer if input/output sizes differ or stride > 1.
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # First block with optional downsampling.
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer)
        )
        self.inplanes = planes * block.expansion  # Update inplanes for the next blocks.

        # Add the remaining blocks without downsampling.
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                      dilation=self.dilation, norm_layer=norm_layer)
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        Implementation of the forward pass.

        Args:
            x (Tensor): Input tensor (batch of images).

        Returns:
            Tensor: Model output (logits for each class).
        """
        # Initial convolution + BN + ReLU + Max Pooling.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Pass through the four ResNet layers.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, feature_dim).

        # Final classification layer.
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass wrapper.

        Args:
            x (Tensor): Input tensor (batch of images).

        Returns:
            Tensor: Model output (logits for each class).
        """
        return self._forward_impl(x)

def _resnet(
    block: Type[Bottleneck],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:
    """
    Constructs a ResNet model using the specified block type and layer configuration.

    Args:
        block (Type[Bottleneck]): The type of residual block to use (e.g., Bottleneck).
        layers (List[int]): A list of integers defining the number of blocks in each layer.
                              For example, [3, 4, 6, 3] corresponds to ResNet-50.
        **kwargs (Any): Additional keyword arguments that are passed to the ResNet constructor.
    
    Returns:
        ResNet: An instance of the ResNet model configured with the given block and layer settings.
    """
    model = ResNet(block, layers, **kwargs)
    return model

def resnet50(**kwargs: Any) -> ResNet:
    """
    Constructs a ResNet-50 model.

    This function creates a ResNet-50 model by calling the internal _resnet function with the Bottleneck block
    and the layer configuration [3, 4, 6, 3].

    Args:
        **kwargs (Any): Additional keyword arguments to be passed to the ResNet constructor.
    
    Returns:
        ResNet: A ResNet-50 model instance.
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

def create_resnet(num_classes: int, model_name: str = "resnet50", dropout: float = 0.2):
    """
    Factory function to create a ResNet model for classification with a specified number of classes.

    Args:
        num_classes (int): The number of classes for the classification task.
        model_name (str, optional): The model variant to create. Currently, only "resnet50" is supported.
                                    Defaults to "resnet50".
        dropout (float, optional): Dropout rate (provided for compatibility, but not used in this function).
                                   Defaults to 0.2.
    
    Returns:
        ResNet: The constructed ResNet model configured for the given number of classes.
    
    Raises:
        ValueError: If an invalid model name is provided.
    """
    if model_name == "resnet50":
        model = resnet50(num_classes=num_classes)
    else:
        raise ValueError("Invalid model name.")
    return model

def create_resnet_cifar100_hf():
    """
    Creates a ResNet-50 model pre-trained on CIFAR-100 from the HuggingFace Hub.

    This function checks if the pre-trained model weights exist locally; if not, it downloads them using
    hf_hub_download. It then creates the model using timm, modifies the first convolution layer to better
    suit CIFAR-100 image dimensions (3x3 kernel, stride 1, padding 1), and finally loads the pretrained
    state dictionary.

    Returns:
        torch.nn.Module: A ResNet-50 model pre-trained on CIFAR-100.
    """
    # Check if the model weights file exists locally; if not, download it.
    if not os.path.isfile("./data/models/resnet50_cifar100_hf/pytorch_model.bin"):
        hf_hub_download(
            repo_id="edadaltocg/resnet50_cifar100",
            filename="pytorch_model.bin",
            local_dir="./data/models/resnet50_cifar100_hf"
        )
    # Create the model using timm. 'pretrained=False' because we'll load our own weights.
    model = timm.create_model("hf_hub:edadaltocg/resnet50_cifar100", pretrained=False)
    
    # Modify the first convolutional layer to use a 3x3 kernel with stride 1 and padding 1,
    # which is typically preferred for CIFAR-100 images.
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    
    # Load the pre-trained weights from the local file.
    model.load_state_dict(torch.load("./data/models/resnet50_cifar100_hf/pytorch_model.bin"))
    
    return model

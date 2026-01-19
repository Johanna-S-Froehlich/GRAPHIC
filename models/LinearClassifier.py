import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, backbone, final_layer, feature_dimension, output_dim, freeze_backbone=True):
        """
        Attaches a linear classifier to a pre-trained backbone model at a certain layer and discards the rest of the model.
        Args:
            backbone (nn.Module): The pre-trained model (e.g., a CNN or another architecture).
            final_layer (str): The name of the final layer in the backbone model where the feature extraction ends.
            feature_dimension (int): The number of features that will be passed into the classifier (usually the output size of the 'final_layer').
            output_dim (int): The number of output classes for the linear classifier (e.g., 10 for a 10-class classification task).
            freeze_backbone (bool): Whether to freeze the parameters of the backbone (default is True, meaning no training of backbone parameters).
        """
        # Call the parent class constructor to initialize the module
        super(LinearClassifier, self).__init__()

        # Find the index of the final layer of the backbone where feature extraction ends.
        # `list(backbone.children())` gives all the child layers of the backbone model
        final_layer_idx = list(backbone.children()).index(getattr(backbone, final_layer))

        # Create a sequential container `self.features` with the layers of the backbone up to and including the final layer
        # `list(backbone.children())[:final_layer_idx+1]` slices the list of layers to include everything up to the final layer.
        self.features = nn.Sequential(*list(backbone.children())[:final_layer_idx + 1])

        # Define the classifier layer, which is a fully connected (linear) layer
        # It takes as input the `feature_dimension` (output size from the final layer of the backbone) and outputs `output_dim` classes.
        self.classifier = nn.Linear(in_features=feature_dimension, out_features=output_dim)

        # Optionally freeze the backbone's parameters so that they are not updated during training.
        if freeze_backbone:
            # Set `requires_grad = False` for all parameters in the `features` part of the model.
            # This prevents the model from updating the backbone weights during backpropagation.
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the model:
        - Extract features from the backbone.
        - Flatten the features and pass them through the classifier to get the final output.

        Args:
            x (torch.Tensor): Input tensor with the shape (batch_size, channels, height, width), depending on the backbone.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim), which corresponds to the classification predictions.
        """
        # Pass input through the feature extractor (the backbone part of the model)
        x = self.features(x)

        # Flatten the output from the features. `x.size(0)` is the batch size, and `-1` flattens the rest of the dimensions.
        x = x.view(x.size(0), -1)

        # Pass the flattened features through the classifier to get the output for classification
        x = self.classifier(x)

        return x  # The output will have shape (batch_size, output_dim)

import torchvision.transforms as T


def cifar_transforms(args):
    """
    Defines and returns CIFAR-10 data augmentation and normalization pipelines.

    Args:
        args: An argument object with the following attributes:
            - autoaugment (bool): Whether to apply AutoAugment for stronger data augmentation.

    Returns:
        tuple: (transform_train, transform_test) - Data augmentation and normalization transforms for training and testing.
    """
    # Initialize the training transform list
    transform_train = []

    # Standard data augmentation: Random cropping and horizontal flip
    transform_train.extend([
        T.RandomCrop(32, padding=4),  # Randomly crop a 32x32 region with a padding of 4 pixels
        T.RandomHorizontalFlip(),     # Randomly flip the image horizontally
    ])

    # Apply AutoAugment if specified (uses the CIFAR-10 augmentation policy)
    if args.autoaugment:
        transform_train.append(T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10))

    transform_train.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])  # Normalization for CIFAR-10 RGB
    ])

    # Combine all the training transforms using Compose
    transform_train = T.Compose(transform_train)

    # Define the test-time transforms (no random augmentations, only preprocessing)
    transform_test = []

    # Convert test images to tensor and normalize
    transform_test.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Combine all the test transforms using Compose
    transform_test = T.Compose(transform_test)

    return transform_train, transform_test


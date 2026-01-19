from models.resnet import create_resnet
import torch


def register_hooks(model, activation_dict=None, exclude=None, only=None):
    modules = dict(model.named_modules()) if exclude is None else {k: v for k, v in dict(model.named_modules()).items() if k not in exclude}
    modules = modules if only is None else {k: v for k, v in modules.items() if k in only}
    if activation_dict is None:
        activation_dict = {}

    for layer_name, layer in modules.items():
        layer.register_forward_hook(create_extractor_hook(layer_name, activation_dict))

    return activation_dict, modules


def create_extractor_hook(layer_name, activation_dict):
    def hook_fn(module, input, output):
        activation_dict[layer_name] = output

    return hook_fn


def get_leaf_layers(model, parent_name='', leaf_layers=None):
    if leaf_layers is None:
        leaf_layers = {}

    for name, layer in model.named_children():
        current_name = f"{parent_name}.{name}" if parent_name else name

        # If the layer has no children, add it to the dictionary
        if len(list(layer.children())) == 0:
            leaf_layers[current_name] = layer

        # Recursively call the function for nested layers
        get_leaf_layers(layer, current_name, leaf_layers)

    return leaf_layers


def capture_output_dims(model, flattened=True, input_size=(3, 32, 32), only=None, exclude=None, device='cuda'):
    """
    Captures the output dimensions of all modules in the model.
    :param model: The model to capture the output dimensions from.
    :param input_size: The input size to the model.
    :return: A dictionary containing the output dimensions of all modules in the model.
    """
    modules = dict(model.named_modules()) if exclude is None else {k: v for k, v in dict(model.named_modules()).items()
                                                                   if k not in exclude}
    modules = modules if only is None else {k: v for k, v in modules.items() if k in only}
    x = torch.randn(1, *input_size).to(device)
    output_dims = {}
    hooks = []
    for name, module in modules.items():
        hooks.append(module.register_forward_hook(lambda self, input, output, name=name: output_dims.update({name: torch.flatten(output, 1).shape[1] if flattened else output.shape})))
    model(x)
    for hook in hooks:
        hook.remove()
    return output_dims


def main():
    model = create_resnet(100, 'resnet50').eval()
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    named_modules = dict(model.named_modules()).keys()
    named_leaf_layers = get_leaf_layers(model).keys()

    output_dims = capture_output_dims(model, input_size=(3, 32, 32), only=['layer1', 'layer2', 'layer3', 'layer4'], flattened=True)

    print(output_dims)


if __name__ == '__main__':
    main()

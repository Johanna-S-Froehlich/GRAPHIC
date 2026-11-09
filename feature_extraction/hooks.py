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

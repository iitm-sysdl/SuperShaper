import gc
import torch

# taken from https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/26
def wipe_memory(optimizer):  # DOES WORK
    _optimizer_to(optimizer, torch.device("cpu"))
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()


def _optimizer_to(optimizer, device):
    # when wrapping the optimizer with accelerator the optimizer.state is
    # converted to 'AcceleratorState'
    # So instead of accessing optimizer state with optimizer.state
    # we use optimizer.state_dict()["state"]
    for param in optimizer.state_dict()["state"].values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
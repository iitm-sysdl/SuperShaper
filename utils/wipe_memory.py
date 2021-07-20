import gc
import torch

import subprocess as sp
import os

# taken from https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
def get_gpu_memory():
    _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


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


def free_memory(accelerator, model, optimizer):
    """
    Will release all references to the internal objects stored and call the garbage collector. You should call this
    method between two trainings with different models/optimizers.
    """
    accelerator.print("Wipping gpu memory")
    accelerator.print(f"Free gpu Memory before wiping on each gpus: {get_gpu_memory()}")
    accelerator._optimizers = []
    accelerator._models = []
    accelerator.deepspeed_engine = None
    del model
    wipe_memory(optimizer)
    accelerator.print(f"Free gpu Memory after wiping on each gpus: {get_gpu_memory()}")

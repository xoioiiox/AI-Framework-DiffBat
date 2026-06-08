import mindspore as ms


def configure_runtime(device):
    """Configure MindSpore device for both old and new MindSpore APIs."""
    if hasattr(ms, "set_device"):
        ms.set_context(mode=ms.PYNATIVE_MODE)
        ms.set_device(device)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=device)

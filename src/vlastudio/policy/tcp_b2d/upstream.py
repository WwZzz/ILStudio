"""Load pinned upstream TCP code without redistributing a modified source copy."""
import importlib
import sys
import threading

_LOCK = threading.RLock()


def modules(download=True):
    from vlastudio.utils.driving_assets import source
    root = source('tcp', download)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module('TCP.model'), importlib.import_module('TCP.config')


def network(download=True):
    module, configuration = modules(download)
    # The complete driving checkpoint supplies its backbone. Random initialization
    # also stays offline: upstream's unconditional ImageNet download is redundant.
    with _LOCK:
        original = module.resnet34
        def backbone(*args, **kwargs):
            kwargs['pretrained'] = False
            return original(*args, **kwargs)
        module.resnet34 = backbone
        try:
            result = module.TCP(configuration.GlobalConfig())
        finally:
            module.resnet34 = original
    return result

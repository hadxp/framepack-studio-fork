"""Keep diffusers 0.33 importable with torchao releases that moved dtype classes.

diffusers registers torchao tensor classes during import. On torchao >= 0.17 some of
those modules were renamed, and the diffusers helper then references a logger that
has not been created yet, which aborts startup. Re-export the names it looks up
before the first diffusers import. Missing classes are stubs used only as safe
globals; NVFP4 quantization does not load those checkpoints.
"""

import sys
import types


_INSTALLED = False


def _module(name: str) -> types.ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    module = types.ModuleType(name)
    module.__path__ = []
    parent_name, _, child = name.rpartition(".")
    if parent_name:
        parent = _module(parent_name)
        setattr(parent, child, module)
    sys.modules[name] = module
    return module


def _set_if_missing(module: types.ModuleType, attr: str, value) -> None:
    if not hasattr(module, attr):
        setattr(module, attr, value)


def install_torchao_diffusers_compat() -> None:
    """Make the dtype paths diffusers 0.33 imports exist on current torchao."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    try:
        import torchao.dtypes as dtypes
    except ImportError:
        return

    if not hasattr(dtypes, "NF4Tensor"):
        nf4 = None
        for path in (
            "torchao.quantization.quantize_.workflows.nf4.nf4_tensor",
            "torchao.dtypes.nf4tensor",
        ):
            try:
                nf4 = __import__(path, fromlist=["NF4Tensor"]).NF4Tensor
                break
            except ImportError:
                continue
        _set_if_missing(dtypes, "NF4Tensor", nf4 or type("NF4Tensor", (), {}))

    try:
        from torchao.dtypes.floatx.float8_layout import Float8AQTTensorImpl  # noqa: F401
    except ImportError:
        layout = _module("torchao.dtypes.floatx.float8_layout")
        _set_if_missing(layout, "Float8AQTTensorImpl", type("Float8AQTTensorImpl", (), {}))

    try:
        from torchao.dtypes.uintx.uint4_layout import UInt4Tensor  # noqa: F401
    except ImportError:
        uint4 = _module("torchao.dtypes.uintx.uint4_layout")
        _set_if_missing(uint4, "UInt4Tensor", type("UInt4Tensor", (), {}))

    try:
        from torchao.dtypes.uintx.uintx_layout import (  # noqa: F401
            UintxAQTTensorImpl,
            UintxTensor,
        )
    except ImportError:
        uintx = _module("torchao.dtypes.uintx.uintx_layout")
        real_uintx = None
        try:
            real_uintx = __import__(
                "torchao.prototype.dtypes.uintx.uintx_utils",
                fromlist=["UintxTensor"],
            ).UintxTensor
        except ImportError:
            pass
        _set_if_missing(uintx, "UintxTensor", real_uintx or type("UintxTensor", (), {}))
        _set_if_missing(uintx, "UintxAQTTensorImpl", type("UintxAQTTensorImpl", (), {}))

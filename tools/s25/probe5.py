import sys
import triton.language.core as core
_orig = core._type_for_tuple_values
def _patched(values, fields=None):
    values = [core.constexpr(x) if isinstance(x, list) else x for x in values]
    return _orig(values, fields)
core._type_for_tuple_values = _patched
sys.path.insert(0, "tools/s25")
import probe3  # noqa

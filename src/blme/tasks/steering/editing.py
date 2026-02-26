from ...registry import register_task
from ..gem.editing import MixtureEditingTask as _GemMixtureEditingTask


@register_task("steering_editing")
class MixtureEditingTask(_GemMixtureEditingTask):
    """Backward-compatible alias for GEM mixture editing."""
    pass

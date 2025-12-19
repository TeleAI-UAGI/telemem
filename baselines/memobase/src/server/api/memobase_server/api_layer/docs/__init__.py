from .basic_docs import API_X_CODE_DOCS

# Import all functional documentation modules to populate API_X_CODE_DOCS
from . import project
from . import user
from . import blob
from . import profile
from . import event

__all__ = ["API_X_CODE_DOCS"]

class ITELServingError(Exception):
    """Base error for serving failures."""


class ValidationError(ITELServingError):
    """Raised when the incoming audio does not meet requirements."""


class ModelLoadError(ITELServingError):
    """Raised when model artifacts cannot be loaded."""

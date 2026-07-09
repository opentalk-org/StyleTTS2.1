"""Stdout/stderr capture lives in ``runflow`` as a generic runtime capability. Re-exported here so
callers that already import from ``shared`` keep working."""
from runflow.runtime.log_capture import current_output_logger, route_output_to_logger

__all__ = ["current_output_logger", "route_output_to_logger"]

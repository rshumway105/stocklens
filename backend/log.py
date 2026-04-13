"""
Logging compatibility shim.

Provides a `logger` object that works like loguru's logger.
If loguru is installed, uses it directly. Otherwise, falls back
to Python's stdlib logging with a similar interface.

This lets all modules do:
    from backend.log import logger
and get consistent logging regardless of whether loguru is installed.
"""

try:
    from backend.log import logger
except ImportError:
    import logging
    import sys

    # Configure stdlib logging to look like loguru output
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    ))

    _logger = logging.getLogger("stocklens")
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

    class _LoguruCompat:
        """Minimal loguru-compatible wrapper around stdlib logging."""

        def __init__(self, stdlib_logger: logging.Logger):
            self._log = stdlib_logger

        def _format(self, msg: str, args: tuple) -> str:
            """Convert loguru-style {} placeholders to values."""
            try:
                # loguru uses {} not %s
                for arg in args:
                    msg = msg.replace("{}", str(arg), 1)
            except Exception:
                pass
            return msg

        def debug(self, msg: str, *args, **kwargs):
            self._log.debug(self._format(msg, args))

        def info(self, msg: str, *args, **kwargs):
            self._log.info(self._format(msg, args))

        def warning(self, msg: str, *args, **kwargs):
            self._log.warning(self._format(msg, args))

        def error(self, msg: str, *args, **kwargs):
            self._log.error(self._format(msg, args))

        def critical(self, msg: str, *args, **kwargs):
            self._log.critical(self._format(msg, args))

    logger = _LoguruCompat(_logger)  # type: ignore

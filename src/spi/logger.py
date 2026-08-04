import logging
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"


def configure_logging(logs_dir: Path | None = None) -> Path:
    """Configure application logging once and return the created log-file path."""
    target_dir = logs_dir or Path.cwd() / "logs"
    target_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = target_dir / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    logging.basicConfig(
        filename=log_file_path,
        encoding="utf-8",
        format=LOG_FORMAT,
        level=logging.INFO,
    )
    return log_file_path


if __name__ == "__main__":
    configure_logging()
    logging.getLogger(__name__).info("Logging has started")

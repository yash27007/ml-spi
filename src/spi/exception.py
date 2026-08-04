import logging
import sys

logger = logging.getLogger(__name__)


def error_message_detail(error: Exception, error_detail: object) -> str:
    """Return an error message with the source location of the active exception."""
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return f"Error occurred in Python script: {error}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    return (
        "Error occurred in Python script:\n"
        f"script name: [{file_name}]\n"
        f"line number: [{exc_tb.tb_lineno}]\n"
        f"error message: [{error}]"
    )


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: object) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message


if __name__ == "__main__":
    from spi.logger import configure_logging

    configure_logging()
    try:
        1 / 0
    except Exception as error:
        logger.error("Divide by zero error")
        raise CustomException(error, sys) from error

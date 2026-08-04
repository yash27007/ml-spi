"""Application entry point."""

import logging

from spi.logger import configure_logging


def main() -> None:
    configure_logging()
    logging.getLogger(__name__).info("SPI application started")


if __name__ == "__main__":
    main()

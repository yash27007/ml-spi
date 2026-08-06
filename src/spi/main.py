"""Application entry point."""

from spi.app import create_app
from spi.logger import configure_logging


def main() -> None:
    configure_logging()
    create_app().run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()

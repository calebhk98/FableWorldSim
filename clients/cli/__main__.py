"""Entry point for the CLI client when run as a module."""

from clients.cli.main import main
import sys

if __name__ == "__main__":
    sys.exit(main())

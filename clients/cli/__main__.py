"""Entry point for the CLI client when run as a module."""

import sys

from clients.cli.main import main

if __name__ == "__main__":
    sys.exit(main())

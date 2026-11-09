"""Run the profile tool with python -m speckex."""

from .cli import main

# Delegate command-line parsing and error reporting to the shared entry point.
if __name__ == "__main__":
    main()

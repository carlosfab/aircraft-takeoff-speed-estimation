"""Module for defining file paths and creating directories."""

from pathlib import Path

# Base path of the project
BASE_PATH = Path(__file__).parent.resolve().parent

# Path to the source code directory
CODE_PATH = BASE_PATH / "src"

# Path to the configuration file
CONFIG_PATH = BASE_PATH / "config" / "config.yaml"

# Path to the data directory
DATA_PATH = BASE_PATH / "data"

# Path to the models directory
MODELS_PATH = BASE_PATH / "models"

# Path to the output directory
OUTPUT_PATH = BASE_PATH / "output"


def create_dir_if_not_exists(path: Path) -> None:
    """Create a directory if it does not exist.

    Args:
        path (Path): The path to the directory to create.

    Examples:
        >>> from pathlib import Path
        >>> dir_path = Path("test_dir")
        >>> create_dir_if_not_exists(dir_path)
        Directory 'test_dir' created.
        >>> dir_path.exists()
        True
    """
    if not path.exists():
        path.mkdir(parents=True)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")


def main() -> None:
    """Create directories for the project."""
    create_dir_if_not_exists(DATA_PATH)
    create_dir_if_not_exists(MODELS_PATH)
    create_dir_if_not_exists(OUTPUT_PATH)


if __name__ == "__main__":
    main()

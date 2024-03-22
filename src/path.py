# --- Imports ---
from pathlib import Path

# --- Constants ---
BASE_PATH = Path(__file__).parent.resolve().parent
CONFIG_PATH = BASE_PATH / "config"
DATA_PATH = BASE_PATH / "data"
MODELS_PATH = BASE_PATH / "models"
OUTPUT_PATH = BASE_PATH / "output"


# --- Functions ---
def create_dir_if_not_exists(path: Path) -> None:
    """Create a directory if it does not exist."""
    if not path.exists():
        path.mkdir(parents=True)


def main():
    """Main function."""
    # Create directories if they do not exist
    create_dir_if_not_exists(DATA_PATH)
    create_dir_if_not_exists(MODELS_PATH)
    create_dir_if_not_exists(OUTPUT_PATH)


if __name__ == "__main__":
    main()

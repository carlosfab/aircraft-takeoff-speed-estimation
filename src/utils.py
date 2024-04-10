import argparse
from pathlib import Path

import yaml

from src.path import DATA_PATH, MODELS_PATH, OUTPUT_PATH


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser(description="Speed estimation using YOLOv8")
    ap.add_argument(
        "-v",
        "--video",
        required=False,
        help="Path to input video",
        type=str,
        default=f"{DATA_PATH}/santos_dumont_airport.mp4",
    )
    ap.add_argument(
        "-o",
        "--output",
        required=False,
        help="Path to output video",
        type=str,
        default=f"{OUTPUT_PATH}/output.mp4",
    )
    ap.add_argument(
        "-m",
        "--model",
        required=False,
        help="Path to YOLO model weights",
        type=str,
        default=f"{MODELS_PATH}/yolov8x.pt",
    )

    return ap.parse_args()


def load_config(config_path: Path) -> dict:
    """
    Carrega uma configuração de um arquivo YAML.

    Parâmetros:
    - config_path (Path): O caminho para o arquivo de configuração YAML.

    Retorna:
    - dict: Um dicionário contendo as configurações carregadas do arquivo YAML.

    Levanta:
    - FileNotFoundError: Se o `config_path` não existir.
    - yaml.YAMLError: Se houver um erro ao analisar o arquivo YAML.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Nenhum arquivo de configuração encontrado em {config_path}"
        )

    try:
        with open(config_path, "r") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Erro ao analisar o arquivo YAML: {e}")

    return data

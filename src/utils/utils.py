from pathlib import Path
from datetime import datetime
import logging

def parse_labels(s):
    s = s.strip().strip('"').strip("'")  # retire guillemets autour
    return [label.strip() for label in s.split(",") if label.strip()]

def setup_logger(outdir: str, filename: str):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    log_path = Path(outdir, f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Log also to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    print(f"Logging to {log_path}")
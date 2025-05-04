import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

submodule_path = Path(__file__).parent / "twm" / "external" / "deeplab_forked"
sys.path.append(str(submodule_path))

from twm.external.deeplab_forked.predict import main as deeplab_main  # noqa: E402, I001

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    deeplab_main()

import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

submodule_path = Path(__file__).parent / "twm" / "external" / "deeplab_forked"
sys.path.append(str(submodule_path))

import twm.external.deeplab_forked.main  # noqa: E402
from twm.external.deeplab_forked.main import get_argparser as get_argparser_original  # noqa: E402
from twm.external.deeplab_forked.main import main as deeplab_main  # noqa: E402

if __name__ == "__main__":
    load_dotenv(find_dotenv())

    def get_argparser_wrappere():
        parser = get_argparser_original()
        return parser

    twm.external.deeplab_forked.main.get_argparser = get_argparser_wrappere
    deeplab_main()

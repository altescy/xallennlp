#!/usr/bin/env python
import logging
import os
import sys

from allennlp.commands import main
from allennlp.common.util import import_module_and_submodules

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    level_name = os.environ.get("ALLENNLP_LOG_LEVEL")
    LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL)

_filelock_logger = logging.getLogger("filelock")
_filelock_logger.setLevel(logging.WARNING)


def run():
    import_module_and_submodules("xallennlp")
    main(prog="xallennlp")


if __name__ == "__main__":
    run()

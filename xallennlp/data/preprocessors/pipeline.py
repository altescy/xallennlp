from typing import List

from xallennlp.data.preprocessors.preprocessor import Preprocessor


@Preprocessor.register("pipeline")
class Pipeline(Preprocessor):  # type: ignore[type-arg]
    def __init__(
        self,
        preprocessors: List[Preprocessor],  # type: ignore[type-arg]
    ) -> None:
        self._preprocessors = preprocessors

    def __call__(self, data):  # type: ignore
        for preprocessor in self._preprocessors:
            data = preprocessor(data)
        return data

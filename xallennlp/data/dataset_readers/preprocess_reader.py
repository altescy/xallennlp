from typing import Any, Dict, Iterator

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance

from xallennlp.data.preprocessors import Preprocessor


@DatasetReader.register("preprocess")
class PreprocessReader(DatasetReader):
    def __init__(
        self,
        reader: DatasetReader,
        preprocessors: Dict[str, Preprocessor],  # type: ignore[type-arg]
    ) -> None:
        self._reader = reader
        self._preprocessors = preprocessors
        self._text_to_instance = self._reader.text_to_instance
        self._reader.text_to_instance = self.text_to_instance  # type: ignore

    def _read(self, file_path: str) -> Iterator[Instance]:
        for instance in self._reader._read(file_path):
            yield instance

    def text_to_instance(self, *args: Any, **kwargs: Any) -> Instance:
        kwargs = {key: self._preprocessors[key](value) for key, value in kwargs.items()}
        return self._text_to_instance(*args, **kwargs)  # type: ignore

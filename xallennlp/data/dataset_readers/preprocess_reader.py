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
        **kwargs: Any,
    ) -> None:
        super_kwargs = {
            "max_instances": reader.max_instances,
            "manual_distributed_sharding": reader.manual_distributed_sharding,
            "manual_multiprocess_sharding": reader.manual_multiprocess_sharding,
            "serialization_dir": reader.serialization_dir,
        }
        super_kwargs.update(kwargs)
        super().__init__(**super_kwargs)  # type: ignore
        self._reader = reader
        self._preprocessors = preprocessors
        self._text_to_instance = self._reader.text_to_instance
        self._reader.text_to_instance = self.text_to_instance  # type: ignore

    def _read(self, file_path: str) -> Iterator[Instance]:
        for instance in self._reader._read(file_path):
            yield instance

    def text_to_instance(self, *args: Any, **kwargs: Any) -> Instance:
        kwargs = {
            key: self._preprocessors[key](value) if key in self._preprocessors else value
            for key, value in kwargs.items()
        }
        return self._text_to_instance(*args, **kwargs)  # type: ignore

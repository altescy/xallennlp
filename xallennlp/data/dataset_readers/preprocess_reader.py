from typing import Any, Dict, Iterator

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from xallennlp.data.preprocessors import Preprocessor


@DatasetReader.register("preprocess")
class PreprocessReader(DatasetReader):
    def __init__(
        self,
        base_reader: DatasetReader,
        preprocessors: Dict[str, Preprocessor],  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self.reader = base_reader
        self._preprocessors = preprocessors
        self._text_to_instance = self.reader.text_to_instance
        self.reader.text_to_instance = self.text_to_instance  # type: ignore

        self.reader._set_worker_info(None)
        self.reader._set_distributed_info(None)

    def _read(self, file_path: str) -> Iterator[Instance]:
        for instance in self.reader._read(file_path):
            yield instance

    def text_to_instance(self, *args: Any, **kwargs: Any) -> Instance:
        kwargs = {
            key: self._preprocessors[key](value) if key in self._preprocessors else value
            for key, value in kwargs.items()
        }
        return self._text_to_instance(*args, **kwargs)  # type: ignore

    def apply_token_indexers(self, instance: Instance) -> None:
        self.reader.apply_token_indexers(instance)

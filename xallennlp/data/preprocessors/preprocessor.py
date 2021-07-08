from typing import Generic, TypeVar

from allennlp.common.registrable import Registrable

S = TypeVar("S")
T = TypeVar("T")


class Preprocessor(Generic[S, T], Registrable):
    def __call__(self, data: S) -> T:
        raise NotImplementedError

from xallennlp.data.preprocessors.preprocessor import Preprocessor


@Preprocessor.register("lowercase")
class Lowercase(Preprocessor[str, str]):
    @staticmethod
    def __call__(data: str) -> str:
        return data.lower()

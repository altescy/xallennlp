import numpy.testing
import torch
from xallennlp.modules.seq2seq_encoders.window_concat_encoder import WindowConcatEncoder


def test_window_concat_encder() -> None:
    encoder = WindowConcatEncoder(input_dim=2, window_size=1)
    inputs = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])
    desired = torch.Tensor([[[1, 2, 3, 0, 0, 0, 4, 5, 6], [4, 5, 6, 1, 2, 3, 0, 0, 0]]])

    output = encoder(inputs)
    assert output.size() == (1, 2, 9)
    numpy.testing.assert_almost_equal(output.numpy(), desired.numpy())

    encoder = WindowConcatEncoder(input_dim=2, window_size=(0, 1))
    inputs = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])
    desired = torch.Tensor([[[1, 2, 3, 4, 5, 6], [4, 5, 6, 0, 0, 0]]])

    output = encoder(inputs)
    assert output.size() == (1, 2, 6)
    numpy.testing.assert_almost_equal(output.numpy(), desired.numpy())

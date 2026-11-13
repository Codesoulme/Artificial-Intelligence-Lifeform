import unittest

import torch

from scripts.deepseek_spora_ttt_embedding_probe import fit_predict


class EmbeddingProbeTests(unittest.TestCase):
    def test_source_fit_predicts_linear_mapping(self):
        source = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        outputs = torch.stack((2 * source[:, 0] + 3, -source[:, 1] + 1), dim=-1)
        prediction = fit_predict(source, outputs, torch.tensor([[0.5, 0.25]]), 1e-5)
        torch.testing.assert_close(prediction, torch.tensor([[4., 0.75]]), atol=1e-4, rtol=1e-4)

    def test_predictions_do_not_depend_on_other_test_rows(self):
        source = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
        outputs = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        one = fit_predict(source, outputs, torch.tensor([[0.2, 0.3]]), 0.1)
        batch = fit_predict(source, outputs, torch.tensor([[0.2, 0.3], [100., -50.]]), 0.1)
        torch.testing.assert_close(one[0], batch[0])


if __name__ == '__main__':
    unittest.main()

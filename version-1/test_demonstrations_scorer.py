import json
from unittest.mock import mock_open, patch, MagicMock

import pytest

from demonstrations_scorer import DemonstrationsScorer


@pytest.fixture
def mock_ranker():
    ranker = DemonstrationsScorer()
    ranker.output_file = "test_output/"
    ranker.accelerator = MagicMock()
    ranker.accelerator.device = "cpu"
    return ranker


@patch("DemonstrationsScorer.glob.glob")
@patch("DemonstrationsScorer.logger")
@patch("DemonstrationsScorer.os.remove")
@patch("DemonstrationsScorer.random.shuffle")
@patch("builtins.open", new_callable=mock_open)
@patch("your_module.json.dump")
def test_ranking_candidates(mock_json_dump, mock_open_file, mock_shuffle, mock_os_remove, mock_logger, mock_glob,
                            mock_ranker):

    mock_glob.return_value = ["test_output/tmp_1.bin"]
    fake_ctx = {"ctx": ["ctx1"], "score": 0.9}
    fake_items = [{"idx": 0, "ctx": ["ctx1"], "score": 0.9}]
    mock_open_file.return_value.__enter__.return_value.read.return_value = json.dumps(fake_items)

    with patch("your_module.json.loads", return_value=fake_items), \
            patch("your_module.sorted",
                  side_effect=lambda x, key=None: [(i, {"score": 0.9, "ctxs": "ctx1"}) for i in range(5)]):
        result = mock_ranker.ranking_candidates()

    # Assertions
    assert isinstance(result, list)
    assert "ctx" in result[0]
    assert "ctx_candidates" in result[0]
    mock_json_dump.assert_called_once()
    mock_os_remove.assert_called_once()

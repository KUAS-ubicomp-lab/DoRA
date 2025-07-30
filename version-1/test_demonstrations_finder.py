from unittest.mock import MagicMock, patch

from demonstrations_finder import find_demonstrations


@patch("demonstrations_finder._search", side_effect=lambda args: (args[2], args[2]))
@patch("demonstrations_finder.tqdm.tqdm")  # patch tqdm
@patch("demonstrations_finder.multiprocessing.Pool")
@patch("demonstrations_finder.DemonstrationsFinder")
def test_find_demonstrations(mock_demo_finder_cls, mock_pool_cls, mock_tqdm, mock_search):
    # Mock tqdm context manager
    mock_bar = MagicMock()
    mock_tqdm.return_value.__enter__.return_value = mock_bar
    mock_bar.update.return_value = None

    fake_dataset = [{"text": "sample"} for _ in range(5)]
    fake_task = MagicMock()
    fake_task.dataset = fake_dataset
    fake_task.get_field.side_effect = lambda x: x["text"]

    fake_corpus = MagicMock()
    fake_corpus.task = fake_task
    fake_corpus.is_train = True
    fake_corpus.candidates = ["a", "b", "c"]
    mock_demo_finder_cls.return_value.corpus = fake_corpus

    mock_pool = MagicMock()
    mock_pool.imap_unordered.return_value = [(i, i) for i in range(len(fake_dataset))]
    mock_pool_cls.return_value = mock_pool
    find_demonstrations().retriever = lambda finder: None

    ctx = MagicMock()
    results = find_demonstrations(ctx)

    # Assertions
    assert isinstance(results, list)
    assert len(results) == 5
    for i, demo in enumerate(results):
        assert demo["idx"] == i
        assert demo["ctx_candidates"] == i

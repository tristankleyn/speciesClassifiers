from speciesclassifiers import progress


def test_track_reports_to_hook_and_log(capsys):
    seen = []
    progress.set_hook(lambda what, i, n, eta: seen.append((what, i, n)))
    try:
        assert list(progress.track(range(20), "Reading files")) == list(range(20))
        progress.step("UMAP map")
    finally:
        progress.set_hook(None)
    assert seen[0] == ("Reading files", 0, 20) and seen[-2] == ("Reading files", 20, 20)
    assert seen[-1] == ("UMAP map", 0, None)
    assert "Reading files: 10/20" in capsys.readouterr().out          # the log gets quarters
    assert list(progress.track(iter("abc"), "no length")) == list("abc")  # unknown total, no hook

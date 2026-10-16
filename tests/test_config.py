from contract_assistant.config import ROOT, Settings


def test_relative_cache_dir_is_under_root(monkeypatch):
    monkeypatch.setenv("CACHE_DIR", "somewhere/cache")
    assert Settings().cache_dir == ROOT / "somewhere" / "cache"


def test_absolute_cache_dir_is_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    assert Settings().cache_dir == tmp_path


def test_decode_settings_is_keyed_on_name():
    settings = Settings()
    assert settings.decode_settings()["name"] == settings.extractor_name


def test_decode_settings_has_no_model_path():
    assert set(Settings().decode_settings()) == {
        "name",
        "max_seq_len",
        "doc_stride",
        "max_answer_len",
        "top_k",
        "n_best_size",
    }


def test_extractor_name_from_env(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_NAME", "other-run")
    assert Settings().decode_settings()["name"] == "other-run"

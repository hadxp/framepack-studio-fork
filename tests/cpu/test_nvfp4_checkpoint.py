"""The packed NVFP4 file is created when a load needs it and it is missing."""

from diffusers_helper import nvfp4_checkpoint as nvfp4


def test_packed_file_is_required_when_forced_or_bf16_does_not_fit():
    assert nvfp4.should_use_nvfp4_checkpoint(1, 10, force=True)
    assert nvfp4.should_use_nvfp4_checkpoint(20, 10, force=False)
    assert not nvfp4.should_use_nvfp4_checkpoint(5, 10, force=False)
    assert not nvfp4.should_use_nvfp4_checkpoint(20, None, force=False)


def _patch_loader(monkeypatch, model_dir, weight_bytes, usable):
    monkeypatch.setattr(
        nvfp4, "resolve_model_dir", lambda path, offline=False: model_dir
    )
    monkeypatch.setattr(nvfp4, "checkpoint_nbytes", lambda model_dir: weight_bytes)
    monkeypatch.setattr(nvfp4, "usable_vram_bytes", lambda preservation_gb: usable)
    calls = {"convert": 0, "load": []}

    def convert(directory, force=False):
        calls["convert"] += 1
        path = nvfp4.nvfp4_checkpoint_path(directory)
        path.write_bytes(b"packed")
        return path

    def load(directory, path):
        calls["load"].append(path)
        return "nvfp4-model"

    monkeypatch.setattr(nvfp4, "convert_checkpoint", convert)
    monkeypatch.setattr(nvfp4, "load_nvfp4_transformer", load)
    return calls


def test_missing_packed_file_is_created_then_loaded(tmp_path, monkeypatch):
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    calls = _patch_loader(monkeypatch, model_dir, 30 * 1024**3, 8 * 1024**3)

    loaded = nvfp4.load_hunyuan_transformer(
        "repo", preservation_gb=6, offline=True, force_nvfp4=False
    )

    packed = nvfp4.nvfp4_checkpoint_path(model_dir)
    assert loaded == "nvfp4-model"
    assert calls["convert"] == 1
    assert calls["load"] == [packed]
    assert packed.is_file()


def test_forced_load_creates_a_missing_packed_file(tmp_path, monkeypatch):
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    calls = _patch_loader(monkeypatch, model_dir, 1 * 1024**3, 40 * 1024**3)

    nvfp4.load_hunyuan_transformer(
        "repo", preservation_gb=6, offline=True, force_nvfp4=True
    )

    assert calls["convert"] == 1
    assert calls["load"]


def test_existing_packed_file_is_loaded_without_converting(tmp_path, monkeypatch):
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    packed = nvfp4.nvfp4_checkpoint_path(model_dir)
    packed.write_bytes(b"already")
    calls = _patch_loader(monkeypatch, model_dir, 30 * 1024**3, 8 * 1024**3)

    nvfp4.load_hunyuan_transformer(
        "repo", preservation_gb=6, offline=True, force_nvfp4=True
    )

    assert calls["convert"] == 0
    assert calls["load"] == [packed]
    assert packed.read_bytes() == b"already"


def test_bf16_that_fits_is_not_packed(tmp_path, monkeypatch):
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    calls = _patch_loader(monkeypatch, model_dir, 4 * 1024**3, 40 * 1024**3)
    seen = {}

    class _Model:
        @classmethod
        def from_pretrained(cls, source, torch_dtype=None):
            seen["source"] = source
            return cls()

        def cpu(self):
            return self

        def eval(self):
            return self

        def requires_grad_(self, flag):
            return self

        def apply_nvfp4_w4a16(self):
            raise AssertionError("bf16 that fits should stay bfloat16")

    import sys
    import types

    fake_mod = types.ModuleType("diffusers_helper.models.hunyuan_video_packed")
    fake_mod.HunyuanVideoTransformer3DModelPacked = _Model
    monkeypatch.setitem(
        sys.modules, "diffusers_helper.models.hunyuan_video_packed", fake_mod
    )

    model = nvfp4.load_hunyuan_transformer(
        "repo", preservation_gb=6, offline=True, force_nvfp4=False
    )

    assert isinstance(model, _Model)
    assert seen["source"] == str(model_dir)
    assert calls["convert"] == 0
    assert calls["load"] == []

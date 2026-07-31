"""The restored standalone flow uses POS for heteronyms and NeMo otherwise."""

import importlib.util
import sys
import types
from pathlib import Path


def test_contextual_g2p_routes_heteronyms_through_pos_and_other_words_to_nemo(monkeypatch):
    project_root = Path(__file__).resolve().parent.parent
    pipeline_dir = project_root / "g2p_pipeline_split_v2"
    monkeypatch.syspath_prepend(str(pipeline_dir))

    class FakeIpaG2p:
        pass

    module_names = [
        "nemo",
        "nemo.collections",
        "nemo.collections.tts",
        "nemo.collections.tts.g2p",
        "nemo.collections.tts.g2p.models",
    ]
    for name in module_names:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    leaf = types.ModuleType("nemo.collections.tts.g2p.models.i18n_ipa")
    leaf.IpaG2p = FakeIpaG2p
    monkeypatch.setitem(sys.modules, leaf.__name__, leaf)

    spec = importlib.util.spec_from_file_location(
        "_contextual_g2p_under_test",
        pipeline_dir / "contextual_g2p.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class FakeTagger:
        def __init__(self):
            self.seen = None

        def tag(self, text):
            self.seen = text
            return [("read", "VBP"), ("school", "NN")]

    class FakeNeMo:
        def __init__(self):
            self.seen = []

        def __call__(self, token):
            self.seen.append(token)
            return ["s", "k", "u", "l"]

    tagger = FakeTagger()
    nemo = FakeNeMo()
    pipeline = module.ContextAwareIpaG2p(
        heteronyms_json_path=str(pipeline_dir / "heteronyms.json"),
        ipa_dict_path=str(pipeline_dir / "cmudict-0.7b-ipa.txt"),
        g2p=nemo,
        tagger=tagger,
    )

    assert pipeline("read school") == ["ɹid", "skul"]
    assert tagger.seen == "read school"
    assert nemo.seen == ["school"]

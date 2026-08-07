"""Runtime backend selection must never silently downgrade in service mode."""

import sys
import types

import pytest

from src.core.g2p import g2p_service


class _ResolverWithoutSpacy:
    """Enough resolver surface for backend-selection tests."""


def _make_nemo_import_fail(monkeypatch):
    module_name = "nemo.collections.tts.g2p.models.i18n_ipa"
    monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))
    monkeypatch.setattr(g2p_service, "ContextualHeteronymResolver", _ResolverWithoutSpacy)
    monkeypatch.setattr(g2p_service, "g2p_engine", None)
    monkeypatch.setattr(g2p_service, "g2p_mode", "not_loaded")


def test_missing_nemo_fails_startup_when_required(monkeypatch):
    _make_nemo_import_fail(monkeypatch)
    monkeypatch.setenv("G2P_REQUIRE_NEMO", "1")

    with pytest.raises(RuntimeError, match="NeMo IpaG2p is required"):
        g2p_service.load_g2p_engine()


def test_dictionary_fallback_requires_explicit_opt_out(monkeypatch):
    _make_nemo_import_fail(monkeypatch)
    monkeypatch.setenv("G2P_REQUIRE_NEMO", "0")

    g2p_service.load_g2p_engine()

    assert g2p_service.get_g2p_mode() == "context_aware_dictionary_fallback"

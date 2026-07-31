from __future__ import annotations

import torch
from pathlib import Path

from packages.model_runtime import ModelRegistry
from packages.model_runtime.torch_models import AttentionBiLSTMClassifierV3 as RuntimeModel
from research.src.sequence_model import AttentionBiLSTMClassifierV3 as ResearchModel


def test_czech_runtime_architecture_matches_research_state_dict():
    research_model = ResearchModel(use_layer_mix=True)
    runtime_model = RuntimeModel(use_layer_mix=True)
    runtime_model.load_state_dict(research_model.state_dict(), strict=True)
    research_model.eval()
    runtime_model.eval()
    layers = torch.randn(2, 12, 8, 768)
    mask = torch.ones(2, 8, dtype=torch.bool)

    with torch.inference_mode():
        expected = research_model(layers, mask)
        actual = runtime_model(layers, mask)

    torch.testing.assert_close(actual, expected)


def test_repository_registry_exposes_promoted_models_as_available():
    registry_dir = Path(__file__).resolve().parents[3] / "models" / "registry"
    catalog = {model["id"]: model for model in ModelRegistry(registry_dir).catalog()}

    assert set(catalog) == {"multilingual-cz-en", "czech-layermix"}
    assert catalog["multilingual-cz-en"]["available"]
    assert catalog["czech-layermix"]["available"]


def test_runtimes_share_one_hubert_extractor():
    registry_dir = Path(__file__).resolve().parents[3] / "models" / "registry"
    registry = ModelRegistry(registry_dir)

    czech = registry.get_runtime("czech-layermix")
    multilingual = registry.get_runtime("multilingual-cz-en")

    assert czech.extractor is registry.extractor
    assert multilingual.extractor is registry.extractor
    assert czech.extractor is multilingual.extractor

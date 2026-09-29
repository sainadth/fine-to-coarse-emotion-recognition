from dataclasses import fields
from pathlib import Path

import pytest

from fine2coarse.config import load_yaml, merge_overrides
from fine2coarse.engine import Stage1Config, Stage2Config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_config_maps_onto_its_dataclass(path):
    raw = load_yaml(path)
    dataclass = Stage1Config if path.name == "stage1.yaml" else Stage2Config
    known = {f.name for f in fields(dataclass)}
    assert set(raw) <= known, f"unknown keys: {sorted(set(raw) - known)}"
    dataclass(**raw)


def test_required_systems_are_configured():
    systems = {load_yaml(p).get("system") for p in CONFIG_DIR.glob("*.yaml")}
    assert {"M1", "M2", "M3", "M4"} <= systems


def test_transfer_and_ute_flags_match_the_spec():
    expected = {"M1": (False, False), "M2": (True, False), "M3": (False, True), "M4": (True, True)}
    for path in CONFIG_DIR.glob("*.yaml"):
        raw = load_yaml(path)
        system = raw.get("system")
        if system not in expected:
            continue
        transfer, ute = expected[system]
        assert (raw.get("from_stage1") is not None) is transfer, path.name
        assert bool(raw.get("use_ute")) is ute, path.name


def test_overrides_are_parsed_as_yaml():
    merged = merge_overrides({"epochs": 5}, ["epochs=1", "from_stage1=null", "amp=false"])
    assert merged == {"epochs": 1, "from_stage1": None, "amp": False}

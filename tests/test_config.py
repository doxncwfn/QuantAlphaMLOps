from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from qam import registry
from qam import cli
from qam.config import load_config
from tests.conftest import edit_spec

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["baseline_hybrid", "legacy_parity", "smoke"])
def test_repo_experiments_validate(name):
    cfg = load_config(REPO / "configs" / "experiments" / f"{name}.yaml", env="local")
    assert cfg.name == name
    assert cfg.paths.warehouse.is_absolute()
    assert cfg.filter_spec.func == "topk_dollar_volume"


def test_repo_experiments_differ_only_where_their_headers_say():
    base = load_config(REPO / "configs" / "experiments" / "baseline_hybrid.yaml", env="local").to_plain()
    legacy = load_config(REPO / "configs" / "experiments" / "legacy_parity.yaml", env="local").to_plain()
    differ = {key for key in base if base[key] != legacy[key]}
    assert differ == {"name", "feature_spec", "window_spec", "model_spec"}
    assert base["window_spec"]["transforms"][1:] == legacy["window_spec"]["transforms"][1:]
    smoke = load_config(REPO / "configs" / "experiments" / "smoke.yaml", env="local").to_plain()
    assert {key for key in base if base[key] != smoke[key]} == {"name", "data_spec", "filter_spec", "split_spec", "train_spec"}


def test_cli_time_wins_over_the_spec_with_a_warning(config_file, caplog):
    in_spec = yaml.safe_load(config_file.read_text())["data_spec"]["period"]
    with caplog.at_level(logging.WARNING, logger="qam.config"):
        cfg = load_config(config_file, env=None, times={"start": dt.date(2019, 5, 1), "end": None})
    assert cfg.data_spec.period.start == dt.date(2019, 5, 1)
    assert str(cfg.data_spec.period.end) == in_spec["end"]  # None = not given, spec kept
    assert [r.getMessage() for r in caplog.records] == [f"--start 2019-05-01 overrides data_spec.period.start = {in_spec['start']} from the spec"]


def test_cli_time_equal_to_the_spec_does_not_warn(config_file, caplog):
    same = dt.date.fromisoformat(yaml.safe_load(config_file.read_text())["data_spec"]["period"]["end"])
    with caplog.at_level(logging.WARNING, logger="qam.config"):
        load_config(config_file, env=None, times={"end": same})
    assert not caplog.records


def test_only_time_parameters_come_from_the_cli(config_file):
    with pytest.raises(ValueError, match="cannot be set from the command line"):
        load_config(config_file, env=None, times={"k": 10})
    with pytest.raises(SystemExit):  # argparse rejects the removed generic override
        cli.main(["validate", str(config_file), "--set", "train_spec.max_epochs=1"])


def test_env_overlay_sets_paths(tmp_path, config_file):
    data = yaml.safe_load(config_file.read_text())
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "configs" / "env").mkdir(parents=True)
    (tmp_path / "configs" / "env" / "here.yaml").write_text(yaml.safe_dump({"paths": {**data["paths"], "runs_root": "elsewhere"}}))
    cfg = load_config(config_file, env="here")
    assert cfg.paths.runs_root == (tmp_path / "elsewhere").resolve()


def test_env_overlay_cannot_set_parameters(tmp_path, config_file):
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "configs" / "env").mkdir(parents=True)
    (tmp_path / "configs" / "env" / "sneaky.yaml").write_text(yaml.safe_dump({"train_spec": {"max_epochs": 1}}))
    with pytest.raises(ValueError, match="may only set `paths`"):
        load_config(config_file, env="sneaky")


def test_unknown_component_fails_fast(config_file):
    with pytest.raises(ValueError, match="no model named 'nope'"):
        load_config(edit_spec(config_file, {"model_spec.func": "nope"}, "bad_model"), env=None)


def test_typo_in_key_is_rejected(config_file):
    with pytest.raises(ValidationError):
        load_config(edit_spec(config_file, {"train_spec.max_epoch": 3}, "typo"), env=None)


def test_old_extends_key_is_rejected(config_file):
    with pytest.raises(ValidationError):
        load_config(edit_spec(config_file, {"extends": ["base.yaml"]}, "extends"), env=None)


def test_registry_lists_every_kind():
    for kind in registry.KINDS:
        assert registry.names(kind), f"no {kind} registered"


def test_duplicate_registration_is_an_error():
    with pytest.raises(ValueError, match="already registered"):
        registry.register("loss", "hybrid_hinge")(object())

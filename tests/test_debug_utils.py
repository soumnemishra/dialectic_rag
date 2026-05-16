from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from src.utils.debug_utils import DebugArtifactManager, debug_capture, serialize_for_json


@dataclass
class _SampleDataclass:
    value: int


def test_debug_manager_creates_directories(tmp_path):
    manager = DebugArtifactManager(enabled=True, base_dir=str(tmp_path / "debug"))
    manager.save_text("pubmed/example.txt", "hello")
    assert (tmp_path / "debug" / "pubmed").exists()
    assert (tmp_path / "debug" / "pubmed" / "example.txt").exists()


def test_debug_manager_disabled_no_artifact(tmp_path):
    manager = DebugArtifactManager(enabled=False, base_dir=str(tmp_path / "debug"))
    manager.save_json("retrieval/query.json", {"k": "v"})
    assert not (tmp_path / "debug").exists()


def test_serialize_for_json_supports_dataclass_and_object():
    class Obj:
        def __init__(self):
            self.name = "x"

    payload = {
        "dc": _SampleDataclass(value=7),
        "obj": Obj(),
    }
    serialized = serialize_for_json(payload)
    assert serialized["dc"]["value"] == 7
    assert serialized["obj"]["name"] == "x"


def test_debug_capture_records_exception(tmp_path):
    manager = DebugArtifactManager(enabled=True, base_dir=str(tmp_path / "debug"))

    @debug_capture("failure_case", manager=manager)
    def explode(x: int) -> int:
        raise ValueError(f"bad:{x}")

    with pytest.raises(ValueError):
        explode(5)

    exception_file = tmp_path / "debug" / "failure_case" / "exception.txt"
    assert exception_file.exists()
    assert "bad:5" in exception_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_debug_capture_records_async_io(tmp_path):
    manager = DebugArtifactManager(enabled=True, base_dir=str(tmp_path / "debug"))

    @debug_capture("async_case", manager=manager)
    async def do_work(x: int) -> dict[str, int]:
        return {"value": x}

    result = await do_work(9)
    assert result == {"value": 9}

    inputs_file = tmp_path / "debug" / "async_case" / "inputs.json"
    outputs_file = tmp_path / "debug" / "async_case" / "outputs.json"
    assert inputs_file.exists()
    assert outputs_file.exists()
    saved = json.loads(outputs_file.read_text(encoding="utf-8"))
    assert saved["value"] == 9
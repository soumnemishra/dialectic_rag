import asyncio
import os
from pathlib import Path

import pytest

from run_evaluation import run_evaluation


@pytest.mark.asyncio
async def test_run_one_medqa_question(tmp_path):
    out = await run_evaluation(subset="medqa", limit=1, output_dir=str(tmp_path), verbose=False)
    assert out["summary"].total_questions == 1
    # ensure files were written
    assert Path(out["results_file"]).exists()
    assert Path(out["summary_file"]).exists()

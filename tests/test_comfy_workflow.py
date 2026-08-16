import json
from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "workflows"
    / "minimax_h3_mlx_turbo4_sol.json"
)


def test_workflow_is_self_contained_and_uses_recommended_profile():
    data = json.loads(WORKFLOW.read_text())
    nodes = {node["type"]: node for node in data["nodes"]}
    generator = nodes["MiniMaxH3MLXTurbo"]

    assert generator["widgets_values"][1:6] == [
        "4-bit",
        "Turbo 4 Fast",
        "auto",
        "prequantized 8-bit",
        "sol_attn",
    ]
    assert {"CreateVideo", "SaveVideo", "SaveAudio"}.issubset(nodes)
    assert len(data["links"]) == 4
import json
from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "workflows"
    / "minimax_h3_mlx_turbo4_sol.json"
)
WORKFLOW_24GB = (
    Path(__file__).resolve().parents[1]
    / "workflows"
    / "minimax_h3_mlx_24gb_turbo4_sol.json"
)


def test_workflow_is_self_contained_and_uses_recommended_profile():
    data = json.loads(WORKFLOW.read_text())
    nodes = {node["type"]: node for node in data["nodes"]}
    generator = nodes["MiniMaxH3MLXTurbo"]

    assert generator["widgets_values"][1:6] == [
        "4-bit-pruned",
        "Turbo 4 Fast",
        "auto",
        "prequantized 8-bit",
        "sol_attn",
    ]
    assert generator["widgets_values"][-2:] == [1.3, True]
    assert {"CreateVideo", "SaveVideo", "SaveAudio"}.issubset(nodes)
    assert len(data["links"]) == 4


def test_24gb_workflow_uses_offset_stream2_creator_profile():
    data = json.loads(WORKFLOW_24GB.read_text())
    nodes = {node["type"]: node for node in data["nodes"]}
    generator = nodes["MiniMaxH3MLXGenerate"]

    assert generator["widgets_values"][1:6] == [
        "4-bit-pruned",
        "Turbo 4 Fast",
        "stream2",
        "prequantized 8-bit",
        "sol_attn",
    ]
    assert generator["widgets_values"][-3:] == [1.3, True, "offset"]
    assert generator["widgets_values"][6:10] == [864, 480, 5.0, 42]
    assert {"CreateVideo", "SaveVideo", "SaveAudio", "MarkdownNote"}.issubset(nodes)
    assert "16.36 GB" in nodes["MarkdownNote"]["widgets_values"][0]
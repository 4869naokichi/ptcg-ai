from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ptcg_ai.visualization.render import write_visualizer_html


class VisualizationTest(unittest.TestCase):
    def test_write_visualizer_html_embeds_snapshot_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "replay.html"
            write_visualizer_html(
                snapshots=[{"current": {"turn": 1, "players": []}, "logs": []}],
                output_path=output,
                title="sample",
                metadata={"winner": -1},
            )

            html = output.read_text(encoding="utf-8")
            self.assertIn("sample", html)
            self.assertIn('"turn": 1', html)
            self.assertIn("replay-data", html)
            self.assertIn("playPause", html)


if __name__ == "__main__":
    unittest.main()

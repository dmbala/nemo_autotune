import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "cluster_bench" / "analysis" / "correlate.py"
FIXTURES = ROOT / "tests" / "fixtures" / "cluster_bench" / "history"


class CorrelateTests(unittest.TestCase):
    def _run_correlate(self, fixture_name: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "verdicts.md"
            subprocess.run(
                ["python3", str(SCRIPT), "--history", str(FIXTURES / fixture_name), "--out", str(out)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            return out.read_text()

    def test_degraded_compute_fixture_marks_the_slow_node(self) -> None:
        rendered = self._run_correlate("degraded_compute.jsonl")
        self.assertIn("| nodeA | **DEGRADED_COMPUTE** |", rendered)
        self.assertIn("| nodeB | **OK** |", rendered)

    def test_multiple_fault_fixture_surfaces_multiple_verdict(self) -> None:
        rendered = self._run_correlate("multiple_faults.jsonl")
        self.assertIn("| nodeA | **MULTIPLE:", rendered)
        self.assertIn("DEGRADED_COMPUTE", rendered)
        self.assertIn("DEGRADED_NCCL", rendered)


if __name__ == "__main__":
    unittest.main()

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "cluster_bench" / "scripts" / "counter_delta.sh"
FIXTURES = ROOT / "tests" / "fixtures" / "cluster_bench" / "snapshots"


class CounterDeltaTests(unittest.TestCase):
    def test_traffic_only_delta_is_ok(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                str(FIXTURES / "before_clean.json"),
                str(FIXTURES / "after_clean.json"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[counter_delta] OK", result.stdout)

    def test_error_counter_delta_fails(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                str(FIXTURES / "before_clean.json"),
                str(FIXTURES / "after_error_delta.json"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[counter_delta] FAIL", result.stdout)


if __name__ == "__main__":
    unittest.main()

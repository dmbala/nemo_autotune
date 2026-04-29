import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "cluster_bench" / "scripts" / "affinity_check.sh"
FIXTURES = ROOT / "tests" / "fixtures" / "cluster_bench" / "topology"


class AffinityCheckTests(unittest.TestCase):
    def _run(self, fixture_name: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--snapshot",
                str(FIXTURES / fixture_name),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

    def test_ok_snapshot_exits_zero(self) -> None:
        result = self._run("topo_ok.json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[affinity_check] OK", result.stdout)

    def test_warn_snapshot_exits_three(self) -> None:
        result = self._run("topo_warn.json")
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertIn("[affinity_check] WARN", result.stdout)

    def test_fail_snapshot_exits_one(self) -> None:
        result = self._run("topo_fail.json")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("FAIL", result.stdout)


if __name__ == "__main__":
    unittest.main()

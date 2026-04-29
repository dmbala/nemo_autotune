import json
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "cluster_bench" / "analysis" / "scrape_metrics.py"
FIXTURE_RESULTS = ROOT / "tests" / "fixtures" / "cluster_bench" / "results" / "sample"


class ScrapeMetricsTests(unittest.TestCase):
    def test_sample_results_are_ingested_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "history.jsonl"

            first = subprocess.run(
                ["python3", str(SCRIPT), "--results", str(FIXTURE_RESULTS), "--out", str(out)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("[scrape] appended 5 rows", first.stdout)
            self.assertTrue(out.exists())
            manifest = out.with_suffix(".scraped_manifest")
            self.assertTrue(manifest.exists())

            rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 5)

            counts = Counter(row["kind"] for row in rows)
            self.assertEqual(
                counts,
                Counter(
                    {
                        "training_benchmark": 2,
                        "nccl_perf": 2,
                        "storage_save": 1,
                    }
                ),
            )

            second = subprocess.run(
                ["python3", str(SCRIPT), "--results", str(FIXTURE_RESULTS), "--out", str(out)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("[scrape] appended 0 rows", second.stdout)

            rows_after = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
            self.assertEqual(rows_after, rows)


if __name__ == "__main__":
    unittest.main()

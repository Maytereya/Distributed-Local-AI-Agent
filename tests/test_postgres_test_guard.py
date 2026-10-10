import os
from pathlib import Path
import subprocess
import sys
import unittest


class PostgresTestGuardTests(unittest.TestCase):
    def test_wrong_database_is_rejected_before_loading_django(self):
        script=Path(__file__).parents[1]/"openclaw/deploy/run_reporting_postgres_tests.py"
        for url in ("","postgres://synthetic:CANARY_DB_PASSWORD@localhost/reporting_synthetic",
                    "postgres://synthetic:CANARY_DB_PASSWORD@reporting-review-postgres/live_database"):
            with self.subTest(url=url):
                result=subprocess.run([sys.executable,str(script)],env={**os.environ,"DATABASE_URL":url},capture_output=True,text=True,timeout=5)
                self.assertNotEqual(result.returncode,0)
                self.assertIn("synthetic_database_required",result.stderr)
                self.assertNotIn("django",result.stderr.lower())
                self.assertNotIn("CANARY_DB_PASSWORD",result.stdout+result.stderr)

"""Run synthetic tests on the private scratch DB, using production settings."""
import json
import os
import sys
from urllib.parse import urlsplit

database=urlsplit(os.environ.get("DATABASE_URL",""))
if database.hostname!="reporting-review-postgres" or database.path!="/reporting_synthetic":
    raise RuntimeError("synthetic_database_required")
os.environ["REPORTING_WORKER"]="1"
os.environ["DJANGO_SETTINGS_MODULE"]="config.settings"
sys.path.insert(0,"/app")

import django
from django.conf import settings
settings.CHANNEL_LAYERS={"default":{"BACKEND":"channels.layers.InMemoryChannelLayer"}}
settings.MEDIA_ROOT="/tmp/reporting-synthetic-media"
settings.ALLOWED_HOSTS=["testserver","localhost"]
# Async test threads must release their scratch connections before DB teardown.
settings.DATABASES["default"]["CONN_MAX_AGE"]=0
django.setup()
sys.stdout=sys.__stdout__
sys.stderr=sys.__stderr__
from django.test.runner import DiscoverRunner
if sys.argv[1:] == ["--existing-aggregator"]:
    # The original private tests need a credential fixture, never the live key.
    from pathlib import Path
    key=Path("/tmp/reporting-synthetic-key")
    key.write_text("synthetic-existing-tests-only",encoding="utf-8")
    key.chmod(0o600)
    settings.KNOWLEDGE_API_KEY_FILE=str(key)
    import pytest
    failures=pytest.main(["/private-tests","-q","-o","pythonpath=/app","-o","cache_dir=/tmp/reporting-pytest-cache"])
    label="synthetic_existing_aggregator_tests_exit_code"
elif not sys.argv[1:]:
    failures=DiscoverRunner(verbosity=1,interactive=False).run_tests(["reporting.tests","reporting.test_review"])
    label="synthetic_journal_and_review_tests_failed"
else:
    raise RuntimeError("unsupported_synthetic_test_suite")
sys.__stdout__.write(json.dumps({label:failures})+"\n")
raise SystemExit(bool(failures))

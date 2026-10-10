import json
import sys
import tempfile
import unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parents[1]/'openclaw/monitor'))
from host_metrics_contract import validate_host_metrics
from openclaw_monitor import Monitor


def measurement():
    return {'version':1,'checked_at':datetime.now(timezone.utc).isoformat(),'cpu_count':64,'cpu_percent':10,
        'cpu_temperature':50,'memory_total':1024,'memory_available':512,'filesystem_total':2048,'filesystem_free':512,
        'nvmes':[{'id':0,'temperature':35,'passed':True,'critical_warning':0,'media_errors':0,'percentage_used':1}]}


class HostMetricTests(unittest.TestCase):
    def test_host_projection_rejects_text_identifiers_nonfinite_and_impossible_usage(self):
        for mutate in (lambda d:d['nvmes'][0].update(serial='PRIVATE_CANARY'),lambda d:d.update(cpu_percent=float('nan')),
                       lambda d:d.update(memory_available=2048)):
            data=measurement();mutate(data)
            with self.assertRaises(ValueError): validate_host_metrics(data)

    def test_old_or_missing_hardware_data_never_reports_healthy(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor=Monitor({'data_dir':directory})
            source=SimpleNamespace(stat=lambda:SimpleNamespace(st_size=500),read_text=lambda:json.dumps(data))
            data=measurement()
            with patch('openclaw_monitor.Path',return_value=source):
                self.assertEqual(monitor.check_host_metrics()['status'],'ok')
                data['nvmes'][0]['passed']=None
                self.assertEqual(monitor.check_host_metrics()['status'],'warn')
                data['nvmes'][0]['passed']=False
                self.assertEqual(monitor.check_host_metrics()['status'],'crit')
                data['checked_at']=(datetime.now(timezone.utc)-timedelta(minutes=5)).isoformat()
                self.assertEqual(monitor.check_host_metrics()['status'],'warn')
            monitor.outbox.close()

"""One-shot local host exporter. No network, prompts, process list or logs."""
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path


def cpu_ticks():
    ticks=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    return sum(ticks),ticks[3]+ticks[4]


def collect():
    before=cpu_ticks();time.sleep(1);after=cpu_ticks()
    delta=after[0]-before[0]
    cpu=round(100*(1-(after[1]-before[1])/delta),1) if delta>0 else None
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in {'MemTotal','MemAvailable'}: memory[key]=int(value.split()[0])*1024
    temperatures=[]
    for folder in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            if (folder/'name').read_text().strip() in {'k10temp','coretemp','zenpower'}:
                for source in folder.glob('temp*_input'):
                    value=int(source.read_text().strip())/1000
                    if 0<value<=150: temperatures.append(value)
        except (OSError,ValueError): pass
    disks=[]
    for device in sorted(Path('/sys/class/nvme').glob('nvme*')):
        if not re.fullmatch(r'nvme[0-9]{1,2}',device.name): continue
        row={'id':int(device.name[4:]),'temperature':None,'passed':None,'critical_warning':None,'media_errors':None,'percentage_used':None}
        try:
            response=subprocess.run(['/usr/sbin/smartctl','-a','-j','/dev/'+device.name],capture_output=True,text=True,timeout=10)
            value=json.loads(response.stdout)
            health=value.get('nvme_smart_health_information_log',{})
            row.update(temperature=value.get('temperature',{}).get('current'),passed=value.get('smart_status',{}).get('passed'),
                critical_warning=health.get('critical_warning'),media_errors=health.get('media_errors'),percentage_used=health.get('percentage_used'))
        except (OSError,ValueError,subprocess.TimeoutExpired): pass
        disks.append(row)
    usage=shutil.disk_usage('/')
    return {'version':1,'checked_at':datetime.now(timezone.utc).isoformat(),'cpu_count':os.cpu_count(),
        'cpu_percent':cpu,'cpu_temperature':max(temperatures) if temperatures else None,
        'memory_total':memory.get('MemTotal'),'memory_available':memory.get('MemAvailable'),
        'filesystem_total':usage.total,'filesystem_free':usage.free,'nvmes':disks}


if __name__=='__main__':
    from host_metrics_contract import validate_host_metrics
    data=validate_host_metrics(collect())
    target=Path('/home/vl/.reporting-host-metrics/host.json')
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(data,separators=(',',':'))+'\n')
    os.chmod(temp,0o640);os.chown(temp,0,1000);temp.replace(target)

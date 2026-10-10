"""Numeric-only host projection; rejects extra fields including serials."""
import math
from datetime import datetime


def validate_host_metrics(data):
    fields={'version','checked_at','cpu_count','cpu_percent','cpu_temperature','memory_total','memory_available','filesystem_total','filesystem_free','nvmes'}
    if not isinstance(data,dict) or set(data)!=fields or type(data['version']) is not int or data['version']!=1: raise ValueError('invalid_host_metrics')
    if not isinstance(data['checked_at'],str) or len(data['checked_at'])>40 or datetime.fromisoformat(data['checked_at']).tzinfo is None: raise ValueError('invalid_host_metrics')
    limits={'cpu_count':2048,'cpu_percent':100,'cpu_temperature':150,'memory_total':2**63-1,'memory_available':2**63-1,'filesystem_total':2**63-1,'filesystem_free':2**63-1}
    def number(value,maximum,integer=False):
        if value is None: return
        if type(value) not in ((int,) if integer else (int,float)) or not math.isfinite(value) or not 0<=value<=maximum: raise ValueError('invalid_host_metrics')
    for key,limit in limits.items(): number(data[key],limit,key not in {'cpu_percent','cpu_temperature'})
    for total,free in [('memory_total','memory_available'),('filesystem_total','filesystem_free')]:
        if data[total] is not None and data[free] is not None and data[free]>data[total]: raise ValueError('invalid_host_metrics')
    if not isinstance(data['nvmes'],list) or len(data['nvmes'])>16: raise ValueError('invalid_host_metrics')
    ids=set()
    for row in data['nvmes']:
        if not isinstance(row,dict) or set(row)!={'id','temperature','passed','critical_warning','media_errors','percentage_used'}: raise ValueError('invalid_host_metrics')
        if type(row['id']) is not int or not 0<=row['id']<=99 or row['id'] in ids: raise ValueError('invalid_host_metrics')
        ids.add(row['id'])
        if row['passed'] is not None and type(row['passed']) is not bool: raise ValueError('invalid_host_metrics')
        for key,limit in [('temperature',150),('critical_warning',255),('media_errors',2**63-1),('percentage_used',255)]: number(row[key],limit,True)
    return data

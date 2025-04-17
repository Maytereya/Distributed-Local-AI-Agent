import logging


from requests import Session
from requests.adapters import HTTPAdapter, Retry
from functools import lru_cache
from typing import Dict, List, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@dataclass
class DoctorRecord:
    fio: str
    specialization: str
    addresses: List[str]
    short_description: str


class NaykaClient:
    def __init__(self, base_url: str, auth):
        self.base_url = base_url
        self.auth = auth
        self.session = self._init_session()

    def _init_session(self) -> Session:
        sess = Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        sess.mount("https://", HTTPAdapter(max_retries=retries))
        return sess

    def get_json(self, path: str):
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, auth=self.auth, verify=False)
        resp.raise_for_status()
        return resp.json()

    @lru_cache(maxsize=None)
    def get_region_info(self, region_id: int) -> Dict:
        return self.get_json(f"/region/{region_id}")


def build_registry(client: NaykaClient,
                   filter_by_unit: str = None,
                   filter_by_region: str = None) -> List[DoctorRecord]:
    # 1. Загрузить справочники
    doctors = {d['id']: d for d in client.get_json("/doctors")}
    units = {u['id']: u['name'] for u in client.get_json("/companyUnits")}
    regions = client.get_json("/regions")
    region_names = {r['id']: r['name'] for r in regions}
    doc_companies = client.get_json("/doctorCompanyUnits")
    doc_regions = client.get_json("/doctorRegions")

    # 2. Построить worker→regions map
    worker_regions: Dict[int, Set[int]] = {}
    for dr in doc_regions:
        worker_regions.setdefault(dr['worker'], set()).add(dr['region'])

    registry = []
    for entry in doc_companies:
        wid = entry['worker']
        if filter_by_unit and entry['companyUnit'] != filter_by_unit:
            continue

        regions_ids = worker_regions.get(wid, set())
        if filter_by_region and int(filter_by_region) not in regions_ids:
            continue

        addresses = set()
        for rid in regions_ids:
            info = client.get_region_info(rid)
            name = info.get("name") or "не указан"
            addr = info.get("addressForSite") or "не указан"
            if "на дом" in name.lower():
                addresses.add("Этого врача можно вызвать на дом")
            else:
                addresses.add(f"Короткий адрес: {name} | полный адрес: {addr}")  # Было .append пока это был list
        addresses = sorted(set(addresses))

        raw_spec = entry.get("specialization") or ""
        lines = raw_spec.splitlines()
        if lines:
            first_line = lines[0][:150]
        else:
            first_line = ""
        spec = first_line or "Не указано"

        registry.append(DoctorRecord(
            fio=doctors.get(wid, {}).get("fio", "Неизвестный"),
            specialization=units.get(entry["companyUnit"], "–"),
            addresses=addresses,
            short_description=spec,
        ))

    return registry


if __name__ == "__main__":
    import config as c
    import urllib3
    from pprint import pprint

    urllib3.disable_warnings()

    client = NaykaClient(c.nayka_base_url,
                         auth=(c.nayka_login, c.nayka_pass))

    reg = build_registry(client)
    pprint([r.__dict__ for r in reg], width=160)

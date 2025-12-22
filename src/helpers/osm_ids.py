import re

def parse_osm_id_digits(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"(\d{6,12})", s)
    return int(m.group(1)) if m else None
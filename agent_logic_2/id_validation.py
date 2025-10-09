import unicodedata
import re


ID_PATTERN = re.compile(r'^[a-z0-9_]{3,60}$')

RU2LAT = {
    "а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"e","ж":"zh","з":"z","и":"i","й":"i",
    "к":"k","л":"l","м":"m","н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f",
    "х":"h","ц":"c","ч":"ch","ш":"sh","щ":"sch","ъ":"","ы":"y","ь":"","э":"e","ю":"yu","я":"ya"
}
def _ru_to_lat(s: str) -> str:
    out = []
    for ch in s.lower():
        out.append(RU2LAT.get(ch, ch))
    return ''.join(out)

def slugify_ru(text: str) -> str:
    """
    Транслит RU->lat, чистка символов, приведение к [a-z0-9_], сжатие повторов, длина 3..60.
    """
    if not text:
        return ""
    # транслит (если текст на русском — заменится; если латиница — останется)
    t = _ru_to_lat(text)
    # нормализация юникода и нижний регистр
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = t.lower()
    # пробелы/дефисы -> _
    t = re.sub(r'[\s\-]+', '_', t)
    # оставить только a-z0-9_
    t = re.sub(r'[^a-z0-9_]', '', t)
    # сжать повторные _
    t = re.sub(r'_+', '_', t).strip('_')
    # границы длины
    if len(t) < 3:
        t = (t + "_doc")[:3]  # минимально «осмысленная» подпорка
    if len(t) > 60:
        t = t[:60].rstrip('_')
    # пустой случай после чистки
    return t or "doc_id"

def is_valid_id(doc_id: str) -> bool:
    return bool(ID_PATTERN.fullmatch(doc_id or ""))

def sanitize_id(doc_id: str) -> str:
    """Приводит произвольную строку к допустимому ID без участия LLM."""
    return slugify_ru(doc_id)

from __future__ import annotations

from pathlib import Path

# Label - константы
COLLECTIONS_IN_CHROMA = "Коллекции документов Chroma DB"
INDEXES_IN_MEILI = "Индексы документов Meilisearch"

# Static files config for Gradio
PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = (PROJECT_ROOT / "static").resolve()

# Мета-теги для превью в соцсетях
OG_IMAGE_URL = "https://ontheflyai.ru/preview/og.png"
OG_HEAD = (
    "<meta property=\"og:type\" content=\"website\" />\n"
    "<meta property=\"og:title\" content=\"Neiry.ai\" />\n"
    "<meta property=\"og:description\" "
    "content=\"Neiry.ai — чат‑бот для ваших данных. Нажмите, чтобы открыть.\" />\n"
    "<meta property=\"og:url\" content=\"https://ontheflyai.ru/\" />\n"
    "<meta property=\"og:site_name\" content=\"Neiry.ai\" />\n"
    f"<meta property=\"og:image\" content=\"{OG_IMAGE_URL}\" />\n"
    "<meta name=\"twitter:card\" content=\"summary_large_image\" />\n"
    "<meta name=\"twitter:title\" content=\"Neiry.ai\" />\n"
    "<meta name=\"twitter:description\" "
    "content=\"Neiry.ai — чат‑бот для ваших данных. Нажмите, чтобы открыть.\" />\n"
    f"<meta name=\"twitter:image\" content=\"{OG_IMAGE_URL}\" />\n"
)

# -------------------
# Footer/Подвал
# -------------------
custom_css = """

.gradio-container footer {
    display: none !important;
}


/* Шапка с логотипом */
#logo-bar {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  padding: 0 !important;
  margin: 0 !important;           /* убираем нижний отступ */
  border-bottom: none !important;  /* если не нужна линия */
  line-height: 0 !important;       /* убираем «подлипание» снизу из-за baseline */
}

/* Контейнер Row с логотипом — минимальный низ */
#logo-row {
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
}

/* Само изображение: фикс. высота, без кликов, без baseline-отступа */
#brand-logo {
  height: 28px !important;
  width: auto !important;
  display: block !important;       /* убирает baseline-отступ под img */
  pointer-events: none !important; /* без взаимодействия */
  user-select: none !important;
}

/* У верхней кромки табов — убрать отступы */
#main-tabs {
  margin-top: 0 !important;
  padding-top: 0 !important;
}

/* На некоторых версиях Gradio верхняя «полка» табов — отдельный блок */
#main-tabs [data-testid="tab-nav"],
#main-tabs .tab-nav,
#main-tabs .tabs {
  margin-top: 0 !important;
  padding-top: 0 !important;
}
"""


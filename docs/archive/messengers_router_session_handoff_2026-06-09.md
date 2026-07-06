# `messengers_router` — хендофф для новой сессии (2026-06-09)

> **Зачем:** токены прошлой сессии почти кончились. Это durable-точка входа: что сделано,
> что в деплое, что открыто, и **новый топ-приоритет — безопасность (хакерская атака)**.

## 0. ⭐ НОВЫЙ ТОП-ПРИОРИТЕТ: новые баги + ХАКЕРСКАЯ АТАКА → защита

- Владелец сообщил: появились **новые баги** + **хакерская атака**, нужно защититься. **Детали — у владельца в начале сессии** (попросить: что за атака, по каким эндпоинтам/векторам, логи, какие новые баги).
- **Скиллы:** `security-review` (ревью диффа на уязвимости), `engineering:incident-response` (если атака активна — триаж/коммуникация/постмортем), `gsd-secure-phase` (проверка митигаций по threat-model), `superpowers:systematic-debugging` (root cause).
- **Уже есть в дереве:** `tests/test_security_probe.py` (untracked — прочитать, возможно по теме) + eval-стейдж `05_critical_safety_gate` (49 safety-проверок: URGENT/COMPLAINT/MEDICAL_ADVICE/prompt-injection) — это базовая safety-сетка, расширять под атаку.
- **Где смотреть вход:** `messengers_router/endpoint.py` (HTTP-вход `/api/messenger-generate-once` + streaming), nginx перед ботом (403 по source-IP — уже есть фильтр), `early_guards` в `orchestrator.py` (safety-перехват regex до LLM). Атака может быть: prompt-injection в `text`, ресурс-исчерпание (длинный/мульти-тёрн), обход safety, утечка данных через result/PDF-флоу.
- **Принцип:** defense-in-depth (валидация на входе + safety-гейты + rate/size-лимиты), не дезинформировать, не утекать PII (результаты анализов!).

## 1. Что сделано в прошлой сессии (12 коммитов на `release`, `ca0e23e`..`0a7be64`, **НЕ запушено/НЕ задеплоено**)

| Баг | Суть | Коммиты |
|---|---|---|
| **BUG-2026-06-08-01** | результаты: мёртвый getanaliz → портал/PDF interim | `ca0e23e`,`1fbf707` |
| **BUG-2026-06-08-02** (триаж BUG-A) | `/regions` без поля `city` → город из `parent`-иерархии; анализы **1→31** филиал | `88727d8` |
| **BUG-2026-06-08-03** | запись по специальности → doctor-capable (все филиалы), не priceUnits (1) | `a737005` |
| **BUG-2026-06-09-01** (триаж BUG-E ЮГ-2) | толерантный матчинг филиала «ЮГ 2»↔«(ЮГ-2)» | `273d381` |
| триаж **BUG-G** (Калматаева), **BUG-E** Нефтегорск | закрыты BUG-A (филиал не роняется / честный «не по Самаре») | — |

Плюс: диагностик-док адресов (`20fb562`), `current_state §1.13` для Владимира (`0dcc57e`), eval — хардкод Трубина + **динамический шаг 5** (`2a418bd`,`6d00b7d`).

**Прод-факт по результатам (важно):** `resultForPatient` на готовом результате отдаёт **сам PDF (байты `application/pdf`)**, НЕ URL; bot-constructed `getanaliz.php` мёртв (404); портал-форма рабочая, даёт токен-ссылку `…/api/index.php?route=results/patient_download&token=…`. Сейчас бот: готов → портал-interim; не найден/тех-сбой → портал.

## 2. Деплой + верификация (за владельцем)
- **Rebuild нужен** для кода: `a737005` (specialty) + `273d381` (BUG-E ЮГ-2). eval-скрипты — на Mac (просто перезапуск).
- **Верификация после деплоя:** address-probe (гинеколог/кардиолог → много филиалов); E-probe («сдать анализ на ЮГ 2» → филиал ЮГ-2, не 0); eval (динамический шаг + golden ADDRESS зелёный).
- **Для Владимира:** запушить `release` (доки tracked) + прислать `agent_logic_2/config.ini` (в .gitignore, есть `[NAUKA]` секреты → защищённый канал).

## 3. 🔴 Открытые баги из триажа (`messengers_router_bug_triage_2026-06-08.md`)
- **BUG-B**: мульти-лаб-список («ОАК +СОЭ … 30 пунктов») → распознаётся 1, цена игнор.
- **BUG-C**: NLU-мисроутинг («корзина на сайте»→ADDRESS; «срок готовности»→test_assist/result). Чинить **промптом классификатора** (LLM-first, обе dual-location копии), не регексами; прод = `legacy_v2` → чинить и правила.
- **BUG-D**: срок готовности анализа — сперва **ресёрч прод-API** (есть ли поле deadline/days в `api_price`), потом решение.
- **BUG-F**: «надо записываться?» на лаб-анализы → ошибочно предлагает запись (противоречит «анализы без записи»).

## 4. 🧵 Хвосты (не триаж)
- **Результаты — прямая доставка PDF**: ждём (1) Наяку — есть ли эндпоинт, отдающий токен-ссылку (`route=results/patient_download&token`) → бот отдаёт готовую ссылку; (2) иначе — бот стримит PDF-байты (новый `/api/result-pdf`) + агрегатор шлёт файлом (узнать возможность агрегатора). `_extract_result_pdf_url` тогда упростить.
- **ЮГ-2 NLU**: извлекает ли классификатор «ЮГ 2» как branch-сущность (address-фикс уже есть; если NLU не извлекает → отдаст все 31 со ЮГ-2 в списке).
- **FreeTalk-сиблинг результатов** (chip `task_9973f1dd`): `src/localragagent/freetalk/rendering.py` читает старое `result_links`, игнорит `result_preview`. Не на проде.
- **Resilience Phase 2** (`asyncio.gather` параллелизация; план `docs/superpowers/plans/2026-06-06-resilience-degraded-mode-phase1.md`).

## 5. Конвенции / как вести сессию
- Ветка `release` (НЕ переключать). Атомарные коммиты, оканчиваются `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Гейт перед коммитом: `./venv/bin/python -m pytest -q -p no:cacheprovider` + `ruff check`.
- **Скиллы:** на каждый баг `systematic-debugging` (root cause ПЕРЕД фиксом, по живому пробою) → `TDD` (класс-инвариант-тест) → запись в `messengers_router_bug_log.md` → атомарный коммит. Безопасность — `security-review`/`incident-response`.
- **Промпты — в ДВУХ локациях** синхронно (`messengers_router/prompts/*` И `app_data/prompts/mr_*`). NLU прод = `legacy_v2` (rule-path активен). LLM-first NLU (не хрупкие стоп-слова).
- **Прод-пробои:** эндпоинт даёт nginx-403 по source-IP (с Mac И из docker-exec через HTTP). Диагностика — **in-process**: `docker exec bookworm-agent python - <<'PY' … PY` через `api_nayka`/`Services`. Кириллица в paste рвётся на длинных строках → **chunked base64** (строки ≤60). SSH `max@172.16.0.16`.
- **Ключевой прод-факт:** новый бэкенд `medserver-egisz` `/regions` БЕЗ поля `city` (город в дереве `parent` + `companyName`). При region/address/schedule-багах СНАЧАЛА дёрнуть живой эндпоинт, смотреть РЕАЛЬНУЮ структуру (синтетика с выдуманным полем = ложный «код корректен»). Возможны такие же расхождения в `/doctors`, `/priceByRegion`.

## 6. Читать первым (порядок)
1. **этот файл** → 2. `messengers_router_bug_log.md` (BUG-2026-06-08-01/02/03, 06-09-01 — паттерны) → 3. `messengers_router_current_state.md` §1.13 → 4. `messengers_router_address_diagnostic_2026-06-08.md` → 5. `messengers_router_bug_triage_2026-06-08.md` (открытые B/C/D/F).

## 7. Рекомендованный порядок новой сессии
1. **Безопасность** (атака) — получить детали → incident-response/security-review → митигации + тесты. ТОП-приоритет.
2. Новые баги (детали у владельца) — systematic-debugging → TDD.
3. Триаж B/C/D/F (по готовности).

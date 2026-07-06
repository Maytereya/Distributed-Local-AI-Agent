# Процедура: квант ollama-модели + параллелизм (П4 дорожной карты)

> **Кто выполняет:** владелец, на сервере (`max@interrupt`). **Зачем:** сейчас
> `mistral-small3.2:24b-instruct-2506-fp16` (57GB) размазана по 3×RTX 4090 (24GB) —
> межкарточный обмен на каждый токен, LLM-путь медленный, конкурентные пользователи
> сериализуются. Квант q8_0 (~25GB) заметно быстрее; q4_K_M (~14GB) влезает в ОДНУ
> карту и позволяет `OLLAMA_NUM_PARALLEL≥2`. Качество для classify/render на практике
> неотличимо, но это ПРОВЕРЯЕТСЯ, а не предполагается: полный remote eval до/после.
> **Откат — мгновенный** (смена имени модели обратно + restart).

## 0. Baseline ДО изменений (обязательно)

1. С Mac: `./venv/bin/python messengers_router/scripts/measure_latency_baseline.py --reps 3 --out /tmp/lat_before.json`
2. Полный remote eval (`messengers_router/eval_suite/run_remote_eval.sh`) → все стейджи зелёные.
3. На сервере: `docker exec ollama ollama run <текущая fp16> --verbose "Перечисли пять анализов крови"` → записать `eval rate` (tokens/s).

## 1. Скачать квант (не ломает текущую модель)

```bash
docker exec ollama ollama pull mistral-small3.2:24b-instruct-2506-q8_0
# и/или (агрессивнее, для NUM_PARALLEL):
docker exec ollama ollama pull mistral-small3.2:24b-instruct-2506-q4_K_M
```
Если тега q8_0 нет в реестре — `ollama ls`-совместимое имя смотреть на
ollama.com/library/mistral-small3.2/tags (есть q4_K_M и q8_0 у большинства релизов).

## 2. Переключить бота на квант

Имя модели живёт в `agent_logic_2/config.ini` (секция OLLAMA, `ll_model_big` /
`ll_model_small` — вне git, править руками) ЛИБО в рантайм-настройках
`app_data/settings` (ollama options). После правки:
```bash
cd /opt/bookworm_app/Distributed-Local-AI-Agent
docker compose up -d bookworm-agent agent-api   # rebuild не нужен: конфиг хостовый
docker exec ollama ollama ps                    # модель подхватится по 1-му запросу
```

## 3. Параллелизм (только после q4, влезающего в 1 карту)

В compose/env контейнера ollama: `OLLAMA_NUM_PARALLEL=2` (потом 3, если VRAM
позволяет; смотреть `nvidia-smi`). `docker restart ollama` → прогреть запросом.

## 4. Проверка ПОСЛЕ (та же линейка, сравнить с §0)

1. `ollama run <квант> --verbose "..."` → eval rate (ждём заметно выше fp16).
2. С Mac: `measure_latency_baseline.py --reps 3 --out /tmp/lat_after.json` →
   p50/p95 llm-путь должен упасть кратно; rule-путь не изменится.
3. **Полный remote eval** → сравнить со свежим baseline: golden ≥48/49,
   safety 51/51, стейджи 100%. Любая деградация меток/дефлектов → §5 откат.
4. Golden-пробы руками из `docs/baseline_2026-06-26.md`.

## 5. Откат

Вернуть прежнее имя модели в конфиге → `docker compose up -d bookworm-agent
agent-api` → (опц.) `docker exec ollama ollama rm <квант>`. fp16 остаётся на диске.

## Примечания

- Диск: +25GB (q8) / +14GB (q4) на время A/B — проверить `df -h`.
- После любого ребута сервера помнить про GPU-инцидент: `docker exec ollama
  ollama ps` → должно быть ~100% GPU (память `reference_gpu_ollama_cpu_incident`).
- Следующий шаг после успешного q4+parallel (отдельное решение): маленькая модель
  ТОЛЬКО для классификатора (rule=None кейсы) — выше риск, только через eval.

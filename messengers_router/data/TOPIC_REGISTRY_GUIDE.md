# Topic Registry Guide

## Purpose

`topic_registry.yaml` is a single source of truth for routing themes in messenger flow:

- what the user asks about,
- where data should be searched,
- how to mark and classify the topic,
- how to fallback if data is missing.

This file is designed to be editable by humans now and by Gradio UI later.

## File

- Path: `messengers_router/data/topic_registry.yaml`
- Format: YAML
- Encoding: UTF-8

## Topic Object (minimal required fields)

Each topic in `topics` should contain:

1. `topic_id` - stable unique key (snake_case).
2. `enabled` - toggle without deleting topic.
3. `priority` - integer, higher means checked earlier.
4. `label` - router target label (`PREPARE`, `PRICE`, `APPOINTMENT`, etc.).
5. `match` - trigger rules (`any_keywords` / `all_keywords` / `regex` / `exclude_keywords`).
6. `route` - source strategy and source list.
7. `fallback` - action when no relevant data is found.
8. `marks` - analytics metadata (`domain`, `subtype`).

## Strategies

Allowed `route.strategy`:

- `api_only` - only API sources.
- `api_first` - API first, then fallback sources.
- `meili_only` - only Meili index.
- `meili_first` - Meili first, then fallback sources.

## Fallback Actions

Allowed `fallback.action`:

- `handoff` - transfer to operator.
- `clarify` - ask follow-up clarification.
- `reply` - static short reply without transfer.

## Context Controls

`context` fields define behavior if APPOINTMENT flow is already active:

- `interrupts_active_appointment: true|false`
- `require_cancel_confirm_if_appointment_active: true|false`

Recommended default for non-booking topics:

- `interrupts_active_appointment: true`
- `require_cancel_confirm_if_appointment_active: true`

## How To Add A New Topic

1. Copy an existing topic with similar behavior.
2. Set a unique `topic_id`.
3. Adjust `priority` and `label`.
4. Add keywords in `match`.
5. Define source order in `route.sources`.
6. Set explicit `fallback` policy.
7. Keep `marks.domain/subtype` consistent for analytics.

## Priority Rules

- Specific topics must have higher priority than generic topics.
- Example:
  - `prepare_fgds` > `prepare_generic`
  - `doc_tax_certificate` > `doc_generic`

## Gradio Editor (next stage)

For UI editing, expose these operations:

1. list topics
2. create topic
3. update topic
4. enable/disable topic
5. validate YAML schema before save
6. dry-run route preview for test phrase

## Current Scope

Current registry is not yet hard-wired into router logic.
It is an initial canonical dataset for the next integration step.

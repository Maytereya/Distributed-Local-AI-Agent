.PHONY: ft-eval ft-eval-det ft-eval-prod ft-eval-all

ft-eval: ft-eval-det

ft-eval-det:
	python3 tools/freetalk_quality_gate.py --profile deterministic

ft-eval-prod:
	python3 tools/freetalk_quality_gate.py --profile production-like

ft-eval-all:
	python3 tools/freetalk_quality_gate.py --profile deterministic
	python3 tools/freetalk_quality_gate.py --profile production-like

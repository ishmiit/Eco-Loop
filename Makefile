.DEFAULT_GOAL := help
PY := ./.venv/bin/python
export PYTHONPATH := src

.PHONY: help setup doctor run run-fast ablation ecm serve mcp tools runs report test lint clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## venv + deps + EnergyPlus + weather + LLM, then check
	./scripts/setup.sh

doctor:  ## Check EnergyPlus, weather, model and LLM reachability
	$(PY) -m ecoloop doctor

run:  ## Full closed-loop run: 3 days, LLM brain, retrofit pass
	$(PY) -m ecoloop run --days 3 --brain llm --ecm-pass --verbose

run-fast:  ## One simulated day, hourly decisions — quickest honest end-to-end
	$(PY) -m ecoloop run --run-id quick --days 1 --brain llm --decision-interval 60

ablation:  ## Identical run with the deterministic brain (isolates the LLM)
	$(PY) -m ecoloop run --run-id ablation --days 3 --brain heuristic

serve:  ## Live dashboard on http://127.0.0.1:8765
	$(PY) -m ecoloop serve

mcp:  ## MCP server on stdio (for Claude Desktop, IDEs, other agents)
	$(PY) -m ecoloop mcp

tools:  ## Print the tool registry
	$(PY) -m ecoloop tools

runs:  ## List completed runs
	$(PY) -m ecoloop runs

report:  ## Static PNG/PDF charts for the deck, from the newest run
	$(PY) scripts/export_report.py
	$(PY) scripts/update_readme.py

test:  ## Run the test suite
	$(PY) -m pytest tests/ -q

lint:  ## Byte-compile everything as a syntax check
	$(PY) -m compileall -q src tests scripts && echo "syntax ok"

clean:  ## Remove run artifacts (keeps artifacts/examples/)
	find artifacts -mindepth 1 -maxdepth 1 ! -name examples ! -name .gitkeep -exec rm -rf {} +
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	@echo "cleaned"

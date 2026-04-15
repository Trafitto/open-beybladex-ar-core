CORE_DIR   := $(shell pwd)
WEB_DIR    := $(CORE_DIR)/../open_beybladex_ar_web
VENV       := $(CORE_DIR)/.venv/bin/python
WEB_PORT   := 8080
LOW_LIGHT  ?=
_LL        := $(if $(LOW_LIGHT),-l,)

.PHONY: play stop web core help

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-10s %s\n", $$1, $$2}'

play: ## Launch play mode (LOW_LIGHT=1 for -l), press q to stop
	@echo "Starting web server on http://localhost:$(WEB_PORT) ..."
	@cd $(WEB_DIR) && $(VENV) -m http.server $(WEB_PORT) &
	@sleep 0.5
	@echo "Opening Firefox ..."
	@firefox "http://localhost:$(WEB_PORT)" &
	@sleep 0.5
	@echo "Starting core tracker in play mode ..."
	@cd $(CORE_DIR) && $(VENV) main.py -w -p -e $(_LL); \
		echo "Core exited -- shutting down web server ..."; \
		kill %1 2>/dev/null || true

web: ## Start only the web server
	cd $(WEB_DIR) && $(VENV) -m http.server $(WEB_PORT)

core: ## Start only the core tracker (with web + effects)
	cd $(CORE_DIR) && $(VENV) main.py -w -e

stop: ## Kill any leftover background web server
	@-pkill -f "http.server $(WEB_PORT)" 2>/dev/null && echo "Stopped" || echo "Nothing to stop"

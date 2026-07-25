.PHONY: init sync requirements clean

# 1. Initialize a new project structure (creates pyproject.toml if missing)
init:
	uv init

# 2. Sync your environment perfectly with the lockfile
sync:
	uv sync

# 3. Generate a clean requirements.txt file from your uv lockfile
requirements:
	uv export --format requirements-txt -o requirements.txt

# 4. Clean up the virtual environment, lockfile, and generated requirements
clean:
	rm -rf .venv uv.lock requirements.txt

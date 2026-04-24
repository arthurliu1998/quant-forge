#!/bin/bash
set -euo pipefail

TRADE_FORGE_HOME="$HOME/.trade-forge"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== TradeForge Installer ==="

# 1. Create data directory structure
echo "[1/6] Creating data directories..."
mkdir -p "$TRADE_FORGE_HOME"/{data,runs,backups,logs}

# 2. Set up Python virtual environment
echo "[2/6] Setting up Python environment..."
if [ ! -d "$TRADE_FORGE_HOME/.venv" ]; then
    python3 -m venv "$TRADE_FORGE_HOME/.venv"
fi
source "$TRADE_FORGE_HOME/.venv/bin/activate"
pip install -q -r "$REPO_DIR/requirements.txt"

# 3. Copy config templates (don't overwrite existing)
echo "[3/6] Setting up configuration..."
if [ ! -f "$TRADE_FORGE_HOME/.env" ]; then
    cp "$REPO_DIR/.env.example" "$TRADE_FORGE_HOME/.env"
    echo "  Created .env — edit $TRADE_FORGE_HOME/.env to add API keys"
else
    echo "  .env already exists — skipping"
fi

if [ ! -f "$TRADE_FORGE_HOME/config.yaml" ]; then
    cp "$REPO_DIR/config.yaml.example" "$TRADE_FORGE_HOME/config.yaml"
    echo "  Created config.yaml — edit to set your watchlist"
else
    echo "  config.yaml already exists — skipping"
fi

# 4. Set file permissions
echo "[4/6] Setting file permissions..."
chmod 700 "$TRADE_FORGE_HOME"
chmod 600 "$TRADE_FORGE_HOME/.env" 2>/dev/null || true
chmod 600 "$TRADE_FORGE_HOME/config.yaml" 2>/dev/null || true
chmod 700 "$TRADE_FORGE_HOME"/{runs,backups,logs}

# 5. Install git hooks (if in a git repo)
echo "[5/6] Installing git safety hooks..."
if [ -d "$REPO_DIR/.git" ]; then
    HOOK="$REPO_DIR/.git/hooks/pre-commit"
    cat > "$HOOK" << 'HOOKEOF'
#!/bin/bash
# Block sensitive files from being committed
for pattern in ".env" "portfolio.db" "config.yaml" "*.sqlite" "audit.log"; do
    if git diff --cached --name-only | grep -q "$pattern"; then
        echo "ERROR: Attempting to commit sensitive file matching '$pattern'"
        echo "Remove with: git reset HEAD <file>"
        exit 1
    fi
done
HOOKEOF
    chmod +x "$HOOK"
    echo "  Pre-commit hook installed"
else
    echo "  Not a git repo — skipping hooks"
fi

# 6. Check for git history leaks
echo "[6/6] Security checks..."
if [ -d "$REPO_DIR/.git" ]; then
    if git -C "$REPO_DIR" log --all --diff-filter=A --name-only 2>/dev/null | grep -q '\.env$'; then
        echo "  WARNING: .env was previously tracked in git history!"
        echo "  Run: git filter-repo --path .env --invert-paths"
    else
        echo "  No leaked secrets detected in git history"
    fi
else
    echo "  Not a git repo — skipping"
fi

echo ""
echo "=== TradeForge installed ==="
echo "Next steps:"
echo "  1. Edit $TRADE_FORGE_HOME/.env to add API keys"
echo "  2. Edit $TRADE_FORGE_HOME/config.yaml to set your watchlist"
echo "  3. Activate: source $TRADE_FORGE_HOME/.venv/bin/activate"
echo "  4. Test: cd $REPO_DIR && python -m pytest tests/ -v"

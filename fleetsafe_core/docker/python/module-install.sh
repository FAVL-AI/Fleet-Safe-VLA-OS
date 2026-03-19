#!/usr/bin/env bash
# FleetSafeCore Module Install (generic)
# Converts any Dockerfile into a FleetSafeCore module container
#
# Usage in Dockerfile:
#   RUN --mount=from=ghcr.io/dimensionalos/ros-python:dev,source=/app,target=/tmp/d \
#       bash /tmp/d/docker/python/module-install.sh /tmp/d
#   ENTRYPOINT ["/fleetsafe_core/entrypoint.sh"]

set -euo pipefail

SRC="${1:-/tmp/d}"

# ---- Copy source into image (skip if already at /fleetsafe_core/source) ----
if [ "${SRC}" != "/fleetsafe_core/source" ]; then
    mkdir -p /fleetsafe_core/source
    cp -r "${SRC}/fleetsafe_core" "${SRC}/pyproject.toml" /fleetsafe_core/source/
    [ -f "${SRC}/README.md" ] && cp "${SRC}/README.md" /fleetsafe_core/source/ || true
fi

# ---- Find Python + Pip (conda env > venv > uv > system) ----
PYTHON=""
PIP=""

# 1. Check for Conda environment
if [ -z "$PYTHON" ] && command -v conda >/dev/null 2>&1; then
    FLEETSAFE_CORE_CONDA_ENV="${FLEETSAFE_CORE_CONDA_ENV:-app}"
    if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "${FLEETSAFE_CORE_CONDA_ENV}"; then
        PYTHON="conda run --no-capture-output -n ${FLEETSAFE_CORE_CONDA_ENV} python"
        PIP="conda run -n ${FLEETSAFE_CORE_CONDA_ENV} pip"
        echo "Using Conda env: ${FLEETSAFE_CORE_CONDA_ENV}"
    fi
fi

# 2. Check for venv (including uv's .venv)
if [ -z "$PYTHON" ]; then
    for v in /opt/venv /app/venv /venv /app/.venv /.venv; do
        if [ -x "${v}/bin/python" ] && [ -x "${v}/bin/pip" ]; then
            PYTHON="${v}/bin/python"
            PIP="${v}/bin/pip"
            echo "Using venv: ${v}"
            break
        fi
    done
fi

# 3. Check for uv (uses system python but manages deps)
if [ -z "$PYTHON" ] && command -v uv >/dev/null 2>&1; then
    PYTHON="python"
    PIP="uv pip"
    echo "Using uv"
fi

# 4. Fallback to system Python
if [ -z "$PYTHON" ]; then
    PYTHON="python"
    PIP="pip"
    echo "Using system Python"
fi

# ---- Install FleetSafeCore (deps from pyproject.toml[docker]) ----
${PIP} install --no-cache-dir -e "/fleetsafe_core/source[docker]"

# ---- Create entrypoint ----
cat > /fleetsafe_core/entrypoint.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/fleetsafe_core/source:/fleetsafe_core/third_party:\${PYTHONPATH:-}"
exec ${PYTHON} -m fleetsafe_core.core.docker_runner run "\$@"
EOF

chmod +x /fleetsafe_core/entrypoint.sh
echo "FleetSafeCore module installed. Use: ENTRYPOINT [\"/fleetsafe_core/entrypoint.sh\"]"

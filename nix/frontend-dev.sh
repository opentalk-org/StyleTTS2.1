set -euo pipefail

cd src/frontend
if [ ! -d node_modules ]; then
  npm ci
fi

exec npm run dev -- --host 127.0.0.1

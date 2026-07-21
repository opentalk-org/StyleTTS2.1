set -euo pipefail

sync_frontend_dependencies "$PWD/src/frontend"

cd src/frontend
exec npm run dev -- --host 127.0.0.1

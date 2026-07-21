sync_frontend_dependencies() {
  local frontend_dir="$1"
  local package_json="$frontend_dir/package.json"
  local package_lock="$frontend_dir/package-lock.json"
  local vite="$frontend_dir/node_modules/.bin/vite"
  local stamp="$frontend_dir/node_modules/.runflow-npm-ci-stamp"

  test -f "$package_json"
  test -f "$package_lock"

  if [ -x "$vite" ] && [ -f "$stamp" ] \
    && [ ! "$package_json" -nt "$stamp" ] \
    && [ ! "$package_lock" -nt "$stamp" ]; then
    return 0
  fi

  echo "Syncing frontend dependencies from package-lock.json"
  rm -f "$stamp"
  (cd "$frontend_dir" && npm_config_cache="$HOME/.cache/npm" npm ci)
  touch "$stamp"
}

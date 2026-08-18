#!/usr/bin/env bash

# Health-check and restart bgutil-ytdlp-pot-provider when needed.
# Defaults match the paths and proxy used in the current deployment.

set -u

APP_DIR="${APP_DIR:-/usr/local/corpus/jtubespeech/bgutil-ytdlp-pot-provider/server}"
NODE_BIN="${NODE_BIN:-/root/miniconda3/envs/jtubespeech/bin/node}"
PROXY_URL="${PROXY_URL:-http://192.168.8.47:7890}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:4416/ping}"

PID_FILE="${PID_FILE:-${APP_DIR}/bgutil-provider.pid}"
APP_LOG="${APP_LOG:-${APP_DIR}/run.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${APP_DIR}/watchdog.log}"
LOCK_FILE="${LOCK_FILE:-/tmp/bgutil-provider-watchdog.lock}"

export PATH="/root/miniconda3/envs/jtubespeech/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export no_proxy="127.0.0.1,localhost,::1"
export NO_PROXY="${no_proxy}"

log() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" >> "${WATCHDOG_LOG}"
}

if [[ ! -d "${APP_DIR}" ]]; then
    printf '%s ERROR: APP_DIR does not exist: %s\n' "$(date '+%F %T')" "${APP_DIR}" >&2
    exit 1
fi

if [[ ! -x "${NODE_BIN}" ]]; then
    NODE_BIN="$(command -v node 2>/dev/null || true)"
fi

if [[ -z "${NODE_BIN}" || ! -x "${NODE_BIN}" ]]; then
    log "ERROR: node executable not found"
    exit 1
fi

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    exit 0
fi

healthy() {
    curl --noproxy '*' --fail --silent --show-error --max-time 5 \
        "${HEALTH_URL}" >/dev/null 2>&1
}

if healthy; then
    exit 0
fi

log "WARN: health check failed; attempting restart"

# Stop only a process previously started by this script, and verify its
# command line before sending a signal in case the PID has been reused.
if [[ -r "${PID_FILE}" ]]; then
    old_pid="$(<"${PID_FILE}")"
    if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
        old_cmd="$(tr '\0' ' ' < "/proc/${old_pid}/cmdline" 2>/dev/null || true)"
        if [[ "${old_cmd}" == *"build/main.js"* ]]; then
            kill "${old_pid}" 2>/dev/null || true
            for _ in {1..10}; do
                if ! kill -0 "${old_pid}" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "${old_pid}" 2>/dev/null; then
                kill -9 "${old_pid}" 2>/dev/null || true
            fi
        else
            log "WARN: PID ${old_pid} is not bgutil; it was not stopped"
        fi
    fi
    rm -f "${PID_FILE}"
fi

cd "${APP_DIR}" || exit 1
nohup "${NODE_BIN}" build/main.js >> "${APP_LOG}" 2>&1 </dev/null &
new_pid=$!
printf '%s\n' "${new_pid}" > "${PID_FILE}"

sleep 3
if healthy; then
    log "INFO: bgutil provider started successfully with PID ${new_pid}"
    exit 0
fi

log "ERROR: restart failed; inspect ${APP_LOG}"
exit 1

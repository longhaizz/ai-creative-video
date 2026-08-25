// PM2 process file for the bare-metal setup. See server/docs/bare-metal.md.
//
//   pm2 start ecosystem.config.js
//   pm2 logs dub
//   pm2 save && pm2 startup      # survive a reboot
//
// PM2 does not read .env by itself, so this file does it. Values already in
// the real environment win, the same as `set -a && source .env`.
const fs = require('fs')
const path = require('path')

function readEnvFile(file) {
  if (!fs.existsSync(file)) return {}
  const out = {}
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    const m = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/)
    if (!m) continue // comment or blank
    out[m[1]] = m[2].trim().replace(/^(['"])(.*)\1$/, '$2')
  }
  return out
}

const root = __dirname

module.exports = {
  apps: [
    {
      name: 'dub',
      cwd: root,
      script: '/opt/venv-main/bin/uvicorn',
      // One worker on purpose. A second one takes jobs from the same queue
      // and puts two of them on the same card.
      args: 'server.app:app --host 0.0.0.0 --port 8000 --workers 1',
      interpreter: 'none',
      instances: 1,
      autorestart: true,
      restart_delay: 10000,
      // Loading the models takes 1-2 minutes. Without this PM2 counts a slow
      // start as a crash loop.
      min_uptime: 180000,
      max_restarts: 5,
      kill_timeout: 30000,
      env: Object.assign(
        {
          VSR_PYTHON: '/opt/venv-vsr/bin/python',
          OPEN_DUBBING_PYTHON: '/opt/venv-od/bin/python',
          HF_HOME: '/models/huggingface',
        },
        readEnvFile(path.join(root, '.env')),
      ),
    },
  ],
}

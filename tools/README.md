# Tools

## `notion_push_log.py` — log every GitHub push to Notion

Runs from `.github/workflows/notion-on-push.yml` on every push. It creates one
Notion subpage per push with the commit link, message and changed files. If a
results file named `latest.json` changed in that push, it also adds the
experiment's metrics and uploads the PNG figures listed in that file. It never
reuses older results on code-only pushes.

### Setup

1. In Notion, create a parent page (e.g. `PICART — Experiment Log`) and give a
   Notion integration edit access to it.
2. In GitHub, go to **Settings → Secrets and variables → Actions** and add two
   repository secrets:
   - `NOTION_TOKEN`: the integration's API token
   - `NOTION_PARENT_PAGE_ID`: the parent page's ID
3. Push. Check **Actions → PICART Push to Notion** for a green run and a new
   Notion subpage.

Until both secrets are set, the workflow skips logging and still passes.

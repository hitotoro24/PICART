"""GitHub Actions push event -> Notion experiment log child page.

Requires NOTION_TOKEN and NOTION_PARENT_PAGE_ID secrets. No polling.
Experiment results/images are included ONLY when experiments/latest.json changed
in that push, preventing reuse of stale numbers on code-only commits.

This intentionally creates subpages under a Notion *page*, not database rows:
no database schema/setup is required. Real PNGs are uploaded to Notion, so
private repositories do not need publicly reachable image URLs.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import requests

API = "https://api.notion.com/v1"
NOTION_VERSION = "2025-09-03"


def git(*args: str) -> str:
    proc = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def changed_files(sha: str, before: str) -> list[str]:
    if not before or set(before) == {"0"}:
        lines = git("ls-tree", "-r", "--name-only", sha)
    else:
        lines = git("diff", "--name-only", before, sha)
    return [line for line in lines.splitlines() if line]


def rt(value: str) -> list[dict]:
    """Notion rich-text text object list, truncated/split to 2000-char chunks."""
    value = str(value)
    chunks = [value[i:i + 1900] for i in range(0, min(len(value), 7600), 1900)] or [""]
    return [{"type": "text", "text": {"content": s}} for s in chunks]


def paragraph(s: str) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rt(s)}}


def heading(s: str) -> dict:
    return {"object": "block", "type": "heading_2", "heading_2": {"rich_text": rt(s)}}


def request_json(method: str, endpoint: str, token: str, **kwargs) -> dict:
    headers = {"Authorization": f"Bearer {token}", "Notion-Version": NOTION_VERSION}
    if "files" not in kwargs:
        headers["Content-Type"] = "application/json"
    response = requests.request(method, API + endpoint, headers=headers, timeout=45, **kwargs)
    response.raise_for_status()
    return response.json()


def upload_image(path: Path, token: str) -> dict:
    # Notion File Uploads: create -> send (single_part) -> reference ID in block.
    created = request_json("POST", "/file_uploads", token, json={
        "mode": "single_part", "filename": path.name, "content_type": "image/png"})
    file_upload_id = created["id"]
    with path.open("rb") as f:
        sent = request_json("POST", f"/file_uploads/{file_upload_id}/send", token,
                            files={"file": (path.name, f, "image/png")})
    if sent.get("status") not in ("uploaded", "completed"):
        raise RuntimeError(f"Notion file upload did not complete: {sent.get('status')}")
    return {"object": "block", "type": "image", "image": {
        "type": "file_upload", "file_upload": {"id": file_upload_id},
        "caption": rt(path.name)}}


def construct_record(sha: str, repo: str, message: str, changed: list[str]) -> tuple[str, list[dict], dict | None]:
    short = sha[:7]
    url = f"https://github.com/{repo}/commit/{sha}"
    fresh_results = "experiments/latest.json" in changed
    experiment = json.loads(Path("experiments/latest.json").read_text(encoding="utf-8")) if fresh_results else None
    kind = "Experiment" if fresh_results else "Code change"
    title = f"PICART | {short} | {kind}"
    blocks = [
        heading("Version and change"),
        paragraph(f"Commit: {short} ({sha})\nURL: {url}\nRepository: {repo}"),
        paragraph("Commit message: " + message[:1500]),
        paragraph("Changed files:\n" + ("\n".join(changed[:70]) or "(none reported)")),
    ]
    if experiment:
        blocks.extend([
            heading("Experiment"),
            paragraph("ID: " + str(experiment.get("experiment_id", "unrecorded"))),
            paragraph("Question/description: " + str(experiment.get("description", "not provided"))),
            paragraph("Hypothesis: " + str(experiment.get("hypothesis", "not provided"))),
            paragraph("Design: " + str(experiment.get("method", "not provided"))),
            paragraph("Config: " + json.dumps({key: experiment.get(key) for key in ["seed","n","sigma","rate"]})),
            paragraph("Metrics:\n" + json.dumps({"cart_rmse_vs_truth":experiment.get("cart_rmse_vs_truth"),
                "picart_runs":[{k: m.get(k) for k in ["lambda_d", "threshold", "mu_left", "mu_right", "test_rmse_vs_truth", "physics_residual_abs"]}
                        for m in experiment.get("picart_runs", [])[:20]]}, indent=2)),
            paragraph("Interpretation: " + str(experiment.get("interpretation", "not recorded"))),
            paragraph("Next step: " + str(experiment.get("next_step", "not recorded"))),
            heading("Results visuals"),
        ])
    else:
        blocks.append(paragraph("Code-only push; no new experiment result was committed. No older metrics or plots have been reused."))
    return title, blocks, experiment


def main() -> None:
    sha = os.getenv("GITHUB_SHA", "").strip()
    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    event_path = os.getenv("GITHUB_EVENT_PATH", "").strip()
    if not sha or not repo or not event_path:
        raise RuntimeError("Run this script inside a GitHub Actions push event")
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    before = event.get("before", "")
    changes = changed_files(sha, before)
    message = (event.get("head_commit") or {}).get("message", "(not supplied)")
    title, blocks, experiment = construct_record(sha, repo, message, changes)
    token = os.getenv("NOTION_TOKEN", "").strip()
    parent = os.getenv("NOTION_PARENT_PAGE_ID", "").strip()
    if not token or not parent:
        raise RuntimeError("Set GitHub Actions repository secrets NOTION_TOKEN and NOTION_PARENT_PAGE_ID")
    if experiment:
        for relative in experiment.get("figures", [])[:5]:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".png":
                raise ValueError(f"Invalid figure path in latest.json: {relative}")
            if not path.is_file():
                raise FileNotFoundError(f"Expected committed visual missing: {relative}")
            blocks.append(upload_image(path, token))
    created = request_json("POST", "/pages", token, json={
        "parent": {"type": "page_id", "page_id": parent},
        "properties": {"title": {"type": "title", "title": rt(title)}},
        "children": blocks,
    })
    print("Created Notion record:", created.get("url", "(URL not returned)"))


if __name__ == "__main__":
    main()

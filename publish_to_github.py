"""
Publish scanner output to GitHub
================================
Uploads the generated dashboard files to the GitHub repository via the
Contents API so the hosted dashboard picks up fresh data. Authenticates
with a fine-grained personal access token that the script reads from the
environment — the token is NEVER stored in this repository.

Setup
-----
1. Create a fine-grained token at https://github.com/settings/personal-access-tokens
   with access to this repository and the "Contents: Read and write"
   repository permission.
2. Provide it to the script either way:
     * export GITHUB_TOKEN="github_pat_..."          (shell / cron / scheduler)
     * or put GITHUB_TOKEN=github_pat_... in a local .env file next to this
       script (.env is git-ignored — never commit it)
3. To rotate the token later, just replace the value in the same place.

Optional environment overrides:
     GITHUB_REPO    owner/repo   (default: bhatashwani-hash/daily-trading-scanner)
     GITHUB_BRANCH  branch name  (default: the repository's default branch)

Usage:  python india_scanner.py && python sector_scanner.py && python publish_to_github.py
"""

import base64
import os
import sys

import requests

API = "https://api.github.com"
DEFAULT_REPO = "bhatashwani-hash/daily-trading-scanner"
FILES = [
    "india_dashboard.md",
    "india_dashboard.json",
    "sector_table.md",
    "sector_table.json",
]


def load_dotenv(path=".env"):
    """Minimal .env loader — real environment variables take precedence."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            os.environ.setdefault(key, value)


def get_token():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        sys.exit(
            "No GitHub token found. Set the GITHUB_TOKEN environment variable "
            "(or add GITHUB_TOKEN=... to a local .env file) with a fine-grained "
            "token that has Contents: Read and write access to the repo."
        )
    return token


def main():
    load_dotenv()
    token = get_token()
    repo = os.environ.get("GITHUB_REPO", DEFAULT_REPO)
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )

    branch = os.environ.get("GITHUB_BRANCH")
    if not branch:
        resp = session.get(f"{API}/repos/{repo}")
        if resp.status_code == 401:
            sys.exit("GitHub rejected the token (401). Check that it hasn't expired.")
        if resp.status_code == 404:
            sys.exit(
                f"Repo {repo} not found (404). A fine-grained token also returns "
                "404 when it wasn't granted access to this repository."
            )
        resp.raise_for_status()
        branch = resp.json()["default_branch"]

    pushed, skipped = [], []
    for name in FILES:
        if not os.path.exists(name):
            print(f"skip {name}: not generated (run the scanners first)")
            continue
        with open(name, "rb") as fh:
            content = fh.read()

        url = f"{API}/repos/{repo}/contents/{name}"
        current = session.get(url, params={"ref": branch})
        sha = None
        if current.status_code == 200:
            meta = current.json()
            sha = meta["sha"]
            if base64.b64decode(meta.get("content") or "") == content:
                skipped.append(name)
                continue
        elif current.status_code != 404:
            current.raise_for_status()

        payload = {
            "message": f"Update {name}",
            "content": base64.b64encode(content).decode(),
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha
        put = session.put(url, json=payload)
        if put.status_code == 403:
            sys.exit(
                f"Push of {name} forbidden (403) — the token needs the "
                "'Contents: Read and write' repository permission."
            )
        put.raise_for_status()
        pushed.append(name)
        print(f"pushed {name} -> {repo}@{branch}")

    if skipped:
        print("unchanged: " + ", ".join(skipped))
    if not pushed and not skipped:
        sys.exit("Nothing to publish — run the scanners first.")


if __name__ == "__main__":
    main()

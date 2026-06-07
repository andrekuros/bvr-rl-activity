"""Submit a trained model to the online competition server.

Bundles the trained weights (.zip) together with the student's rewards.json into
a single archive and uploads it. The server re-evaluates the weights against the
locked enemy set, so only the model matters for scoring; the rewards file is kept
for transparency.

Usage:
    python -m bvr.submit --name "Team Falcon" --server http://SERVER:8001
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import requests

from .train import DEFAULT_CONFIG_DIR, DEFAULT_MODEL_PATH


def build_submission_bytes(model_path: str, rewards_path: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, arcname="model.zip")
        if os.path.exists(rewards_path):
            zf.write(rewards_path, arcname="rewards.json")
    return buf.getvalue()


def submit(name: str, server: str, model_path: str = DEFAULT_MODEL_PATH,
           rewards_path: str = None) -> dict:
    rewards_path = rewards_path or os.path.join(DEFAULT_CONFIG_DIR, "rewards.json")
    if not os.path.exists(model_path):
        raise SystemExit(f"Model not found: {model_path}. Train one first with `python -m bvr.train`.")
    payload = build_submission_bytes(model_path, rewards_path)
    url = server.rstrip("/") + "/submit"
    resp = requests.post(
        url,
        data={"name": name},
        files={"submission": ("submission.zip", payload, "application/zip")},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Submit a model to the competition server.")
    parser.add_argument("--name", required=True, help="Your team / student name on the leaderboard.")
    parser.add_argument("--server", required=True, help="Competition server URL, e.g. http://host:8001")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    result = submit(args.name, args.server, args.model)
    print("Submission accepted!")
    print(f"  Score (win rate): {result.get('score')}")
    print(f"  Rank: {result.get('rank')} / {result.get('total')}")
    print(f"  Leaderboard: {args.server.rstrip('/')}/")


if __name__ == "__main__":
    main()

"""Upload a run-logs tarball to a HuggingFace dataset repo (arcadia-impact org, per the
project convention; falls back to the personal namespace if org write is denied)."""
import argparse
from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarball", required=True)
    ap.add_argument("--path-in-repo", required=True)
    ap.add_argument("--repo-name", default="sentiment-utility-logs")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--message", default="upload run logs")
    args = ap.parse_args()

    api = HfApi()
    me = api.whoami()["name"]
    for owner in ["arcadia-impact", me]:
        repo = f"{owner}/{args.repo_name}"
        try:
            api.create_repo(repo_id=repo, repo_type="dataset",
                            private=args.private, exist_ok=True)
            api.upload_file(path_or_fileobj=args.tarball, path_in_repo=args.path_in_repo,
                            repo_id=repo, repo_type="dataset", commit_message=args.message)
            print(f"UPLOADED -> https://huggingface.co/datasets/{repo}/blob/main/{args.path_in_repo}")
            return
        except Exception as e:
            print(f"FAILED {repo}: {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit("upload failed for all candidate owners")


if __name__ == "__main__":
    main()

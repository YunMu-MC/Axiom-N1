# GitHub Release Checklist

Project name: Axiom N1

This repository is intended to publish code only. Do not publish local corpora, API generations, model checkpoints, cache folders, downloaded wheels, or `.env` files.

## Before Commit

1. Run a secret/path scan over tracked candidates.
2. Confirm `.gitignore` excludes `data/`, `runs/`, `checkpoints/`, `downloads/`, `.venv/`, `.pip-cache/`, Rust `target/`, and local `.env` files.
3. Keep only `data/README.md` from the data directory.
4. Review provider config files before publishing. Keep real API keys in `.env` or the process environment only.
5. Choose a license before making the repository public.

## Suggested Commands

```powershell
git init -b main
git status --short --ignored
git add .gitattributes .gitignore .env.example README.md pyproject.toml configs docs scripts src tests rust data/README.md
git status --short
git commit -m "Prepare Axiom N1 code release"
git remote add origin https://github.com/<your-account>/axiom-n1.git
git push -u origin main
```

If a large generated file appears in `git status --short`, stop and update `.gitignore` before committing.

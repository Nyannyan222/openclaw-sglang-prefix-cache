# Uploading This Project To GitHub

This repository is currently a local git repository. To upload it to GitHub, use one of the following methods.

## Option A: Use An Existing GitHub Repository

Create an empty GitHub repository in the browser, then copy its URL.

Example:

```text
https://github.com/<your-account>/openclaw-sglang-prefix-cache.git
```

Then run:

```bash
git remote add origin https://github.com/<your-account>/openclaw-sglang-prefix-cache.git
git branch -M main
git add .gitignore README.md bench_sglang_prefix_cache.py docs benchmark_results
git commit -m "Add OpenClaw SGLang prefix cache baseline"
git push -u origin main
```

## Option B: Use GitHub CLI

Install GitHub CLI:

```bash
gh --version
```

Login:

```bash
gh auth login
```

Create and push a repo:

```bash
git branch -M main
git add .gitignore README.md bench_sglang_prefix_cache.py docs benchmark_results
git commit -m "Add OpenClaw SGLang prefix cache baseline"

gh repo create openclaw-sglang-prefix-cache \
  --private \
  --source . \
  --remote origin \
  --push
```

Change `--private` to `--public` if the project should be public.


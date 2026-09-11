# Publish the repository and website

This is a separate public repository. Do not change the research repository's visibility
and do not push its history here.

1. Create an empty public GitHub repository named `xishi404/KVShareArena`.
2. Push this repository's `main` branch to that destination.
3. Open **Settings → Pages → Build and deployment**. Choose **GitHub Actions**.
4. Run **Actions → Publish leaderboard → Run workflow** if the first run preceded
   the Pages setting change.
5. Confirm the deployment succeeds and the site loads without signing in at
   `https://xishi404.github.io/KVShareArena/`.

The Pages workflow deploys only `leaderboard/`. It does not publish data inputs, source
code, or repository metadata as website assets. Those files remain available through
the public repository. There are no secrets to configure.

For later data updates, review the new results first, edit the measured snapshot, run
`python tools/build_site.py`, run the checks, and commit both data files. Deployment does
not contact either HPC or ingest unreviewed experiments.

Reference: [GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

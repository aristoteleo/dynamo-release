import glob
import os
import re
import subprocess
from logging import info, warning
from pathlib import Path
from shutil import copy, copytree, rmtree
from tempfile import TemporaryDirectory
from typing import Dict, ForwardRef, List, Union

from git import Repo

CUR_DIR = Path(__file__).parent

NOTEBOOK_BRANCH = "main"
DYNAMO_NOTEBOOK_PATH_ENV_VAR = "DYNAMO_DOWNLOAD_NOTEBOOKS"


def _download_docs_dirs(repo_url: str) -> None:
    def copy_docs_dirs(repo_path: Union[str, Path]) -> None:
        repo_path = Path(repo_path)
        print("repo path:", repo_path)
        for dirname in ["notebooks", "gallery", "_static"]:
            rmtree(dirname, ignore_errors=True)  # locally re-cloning
            copytree(repo_path / "docs" / "source" / dirname, dirname)

        # copy all rsts in docs/source/*.rst
        for file_path in glob.glob(str(repo_path / "docs" / "source" / "*.rst")):
            print("%s copied to source" % file_path)
            copy(file_path, "./")  # dest: source

    def fetch_remote(repo_url: str) -> None:
        info(f"Fetching notebooks from repo `{repo_url}`")
        with TemporaryDirectory() as repo_dir:
            branch = NOTEBOOK_BRANCH
            repo = Repo.clone_from(repo_url, repo_dir, depth=1, branch=branch)
            repo.git.checkout(branch, force=True)
            copy_docs_dirs(repo_dir)

    def fetch_local(repo_path: Union[str, Path]) -> None:
        info(f"Fetching notebooks from local path `{repo_path}`")
        repo_path = Path(repo_path)
        if not repo_path.is_dir():
            raise OSError(f"`{repo_path}` is not a directory.")
        copy_docs_dirs(repo_path)

    notebooks_local_path = Path(
        os.environ.get(DYNAMO_NOTEBOOK_PATH_ENV_VAR, CUR_DIR.absolute().parent.parent.parent / "notebooks")
    )
    try:
        fetch_local(notebooks_local_path)
    except Exception as e:
        warning(f"read`{notebooks_local_path}` failed, error message: `{e}`. Trying remote")
        require_download = int(os.environ.get(DYNAMO_NOTEBOOK_PATH_ENV_VAR, 1))
        if not require_download:
            info(f"Used downloaded files as set in ENV")
            return

        fetch_remote(repo_url)

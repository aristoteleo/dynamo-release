"""
A version helper in the spirit of versioneer.
Minimalistic and able to run without build step using pkg_resources.
"""

# __version__ is defined at the very end of this file.

import re
import os
import sys
from pathlib import Path
from subprocess import run, PIPE, CalledProcessError
from textwrap import dedent
from typing import NamedTuple, List, Union, Optional
from logging import getLogger


RE_VERSION = r"([\d.]+?)(?:\.dev(\d+))?(?:[_+-]([0-9a-zA-Z.]+))?"
RE_GIT_DESCRIBE = r"v?(?:([\d.]+)-(\d+)-g)?([0-9a-f]{7})(-dirty)?"
ON_RTD = os.environ.get("READTHEDOCS") == "True"

logger = getLogger(__name__)


def match_groups(regex, target):
    match = re.match(regex, target)
    if match is None:
        raise re.error(f"Regex does not match “{target}”. RE Pattern: {regex}", regex)
    return match.groups()


class Version(
    NamedTuple(
        "Version", [("release", str), ("dev", Optional[str]), ("labels", List[str])]
    )
):
    @staticmethod
    def parse(ver):
        release, dev, labels = match_groups(f"{RE_VERSION}$", ver)
        return Version(release, dev, labels.split(".") if labels else [])

    def __str__(self):
        release = self.release if self.release else "0.0"
        dev = f".dev{self.dev}" if self.dev else ""
        labels = f'+{".".join(self.labels)}' if self.labels else ""
        return f"{release}{dev}{labels}"


def get_version_from_dirname(name, parent):
    """Extracted sdist"""
    parent = parent.resolve()
    logger.info(f"dirname: Trying to get version of {name} from dirname {parent}")

    name_re = name.replace("_", "[_-]")
    re_dirname = re.compile(f"{name_re}-{RE_VERSION}$")
    if not re_dirname.match(parent.name):
        logger.info(f"dirname: Failed; Does not match {re_dirname!r}")
        return None

    logger.info("dirname: Succeeded")
    return Version.parse(parent.name[len(name) + 1 :])


def get_version_from_git(parent):
    parent = parent.resolve()
    logger.info(f"git: Trying to get version from git in directory {parent}")
    kw_compat = (
        dict(encoding="utf-8")
        if sys.version_info >= (3, 6)
        else dict(universal_newlines=True)
    )

    try:
        p = run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(parent),
            stdout=PIPE,
            stderr=PIPE,
            check=True,
            **kw_compat,
        )
    except (OSError, CalledProcessError):
        logger.info("git: Failed; directory is not managed by git")
        return None
    if Path(p.stdout.rstrip("\r\n")).resolve() != parent.resolve():
        logger.info(
            "git: Failed; The top-level directory of the current Git repository"
            " is not the same as the root directory of the distribution"
        )
        return None

    p = run(
        [
            "git",
            "describe",
            "--tags",
            "--dirty",
            "--always",
            "--long",
            "--match",
            "v[0-9]*",
        ],
        cwd=str(parent),
        stdout=PIPE,
        stderr=PIPE,
        check=True,
        **kw_compat,
    )

    release, dev, hex_, dirty = match_groups(
        f"{RE_GIT_DESCRIBE}$", p.stdout.rstrip("\r\n")
    )

    labels = []
    if dev == "0":
        dev = None
    else:
        labels.append(hex_)

    if dirty and not ON_RTD:
        labels.append("dirty")

    logger.info("git: Succeeded")
    return Version(release, dev, labels)


def get_version_from_metadata(name: str, parent: Optional[Path] = None):
    logger.info(f"metadata: Trying to get version for {name} in dir {parent}")
    try:
        from pkg_resources import get_distribution, DistributionNotFound
    except ImportError:
        logger.info("metadata: Failed; could not import pkg_resources")
        return None

    try:
        pkg = get_distribution(name)
    except DistributionNotFound:
        logger.info(f"metadata: Failed; could not find distribution {name}")
        return None

    # For an installed package, the parent is the install location
    path_pkg = Path(pkg.location).resolve()
    if parent is not None and path_pkg != parent.resolve():
        msg = f"""\
            metadata: Failed; distribution and package paths do not match:
            {path_pkg}
            !=
            {parent.resolve()}\
            """
        logger.info(dedent(msg))
        return None

    logger.info(f"metadata: Succeeded")
    return Version.parse(pkg.version)


def get_version(package: Union[Path, str]) -> str:
    """Get the version of a package or module

    Pass a module path or package name.
    The former is recommended, since it also works for not yet installed packages.

    Supports getting the version from

    #. The directory name (as created by ``setup.py sdist``)
    #. The output of ``git describe``
    #. The package metadata of an installed package
       (This is the only possibility when passing a name)

    Args:
       package: package name or module path (``…/module.py`` or ``…/module/__init__.py``)
    """
    path = Path(package)
    if not path.suffix and len(path.parts) == 1:  # Is probably not a path
        v = get_version_from_metadata(package)
        if v:
            return str(v)

    if path.suffix != ".py":
        msg = f"“package” is neither the name of an installed module nor the path to a .py file."
        if path.suffix:
            msg += f" Unknown file suffix {path.suffix}"
        raise ValueError(msg)
    if path.name == "__init__.py":
        name = path.parent.name
        parent = path.parent.parent
    else:
        name = path.with_suffix("").name
        parent = path.parent

    return str(
        get_version_from_dirname(name, parent)
        or get_version_from_git(parent)
        or get_version_from_metadata(name, parent)
        or "0.0.0"
    )


__version__ = get_version(__file__)


if __name__ == "__main__":
    print(__version__)

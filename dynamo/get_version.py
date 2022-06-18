"""
A minimalistic version helper in the spirit of versioneer, that is able to run without build step using pkg_resources.
Developed by P Angerer, see https://github.com/flying-sheep/get_version.
"""
# __version__ is defined at the very end of this file.

import os
import re
from pathlib import Path
from subprocess import PIPE, CalledProcessError, run
from typing import List, NamedTuple, Optional, Union

RE_VERSION = r"([\d.]+?)(?:\.dev(\d+))?(?:[_+-]([0-9a-zA-Z.]+))?"
# RE_GIT_DESCRIBE = r"v?(?:([\d.]+)-(\d+)-g)?([0-9a-f]{7})(-dirty)?"
RE_GIT_DESCRIBE = r"v?(?:([\d.]+)-(.+)-g)?([0-9a-f]{7})(-dirty)?"
ON_RTD = os.environ.get("READTHEDOCS") == "True"


def match_groups(regex, target):
    match = re.match(regex, target)
    if match is None:
        raise re.error(f"Regex does not match “{target}”. RE Pattern: {regex}", regex)
    return match.groups()


class Version(NamedTuple):
    release: str
    dev: Optional[str]
    labels: List[str]

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

    re_dirname = re.compile(f"{name}-{RE_VERSION}$")
    if not re_dirname.match(parent.name):
        return None

    return Version.parse(parent.name[len(name) + 1 :])


def get_version_from_git(parent):
    parent = parent.resolve()

    try:
        p = run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(parent),
            stdout=PIPE,
            stderr=PIPE,
            encoding="utf-8",
            check=True,
        )
    except (OSError, CalledProcessError):
        return None
    if Path(p.stdout.rstrip("\r\n")).resolve() != parent.resolve():
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
        encoding="utf-8",
        check=True,
    )

    release, dev, hex_, dirty = match_groups(f"{RE_GIT_DESCRIBE}", p.stdout.rstrip("\r\n"))

    labels = []
    if dev == "0":
        dev = None
    else:
        labels.append(hex_)

    if dirty and not ON_RTD:
        labels.append("dirty")

    return Version(release, dev, labels)


def get_version_from_metadata(name: str, parent: Optional[Path] = None):
    try:
        from pkg_resources import DistributionNotFound, get_distribution
    except ImportError:
        return None

    try:
        pkg = get_distribution(name)
    except DistributionNotFound:
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
        return None

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
        get_dynamo_version()
        or get_version_from_dirname(name, parent)
        or get_version_from_git(parent)
        or get_version_from_metadata(name, parent)
    )


def get_dynamo_version():
    import pkg_resources

    try:
        _package_name = "dynamo-release"
        _package = pkg_resources.working_set.by_key[_package_name]
        version = _package.version
    except KeyError:
        version = "1.0.9"

    return version


def get_all_dependencies_version(display=True):
    """
    Adapted from answer 2 in
    https://stackoverflow.com/questions/40428931/package-for-listing-version-of-packages-used-in-a-jupyter-notebook
    """
    import pandas as pd
    import pkg_resources
    from IPython.display import display

    _package_name = "dynamo-release"
    _package = pkg_resources.working_set.by_key[_package_name]

    all_dependencies = [str(r).split(">")[0] for r in _package.requires()]  # retrieve deps from setup.py
    all_dependencies.sort(reverse=True)
    all_dependencies.insert(0, "dynamo-release")

    all_dependencies_list = []

    for m in pkg_resources.working_set:
        if m.project_name.lower() in all_dependencies:
            all_dependencies_list.append([m.project_name, m.version])

    df = pd.DataFrame(all_dependencies_list[::-1], columns=["package", "version"]).set_index("package").T

    if display:
        pd.options.display.max_columns = None
        display(df)
    else:
        return df


__version__ = get_version(__file__)


if __name__ == "__main__":
    print(__version__)

from typing import Tuple, List

from setuptools import setup, find_packages


def get_dependencies(requirements_file: str) -> Tuple[List[str], List[str]]:
    deps = []
    links = []
    with open(requirements_file) as i:
        for line in i:
            if (line.startswith('--') or line.startswith('#') or line.startswith('pytest') or
                    line.startswith('flake8')):
                continue
            if line.startswith('git') or line.startswith('http'):
                links.append(line.strip())
            else:
                deps.append(line.strip())
    return deps, links


setup(
    name='instance-wise-masker',
    packages=find_packages(exclude="tests"),
    include_package_data=True,
    version=0.1,
    install_requires=get_dependencies('requirements.txt')[0],
    extras_require={"testing": ["pytest>=5.4.1", "pytest-mock==3.1.0", "responses~=0.12.0"]},
)

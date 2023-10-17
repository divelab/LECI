import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    'cilog==1.2.3',
    'gdown>=4.4.0',
    'matplotlib==3.5.2',
    'munch==2.5.0',
    'networkx==2.8',
    'ogb>=1.3.4',
    'pytest==7.1.2',
    'pytest-cov~=3.0',
    'pytest-xdist~=2.5',
    'ruamel.yaml==0.17.21',
    'sphinx>=4.5',
    'protobuf==3.20.1',
    'sphinx-rtd-theme==1.0.0',
    'tensorboard==2.8.0',
    'tqdm==4.64.0',
    'typed-argument-parser==1.7.2',
    'dive-into-graphs',
    'cvxopt>=1.3.0',
    'pynvml>=11.4.1',
    'psutil>=5.9.1',
    'numpy~=1.21.5',
    'setuptools~=59.5.0'
]

setuptools.setup(
    name="graph-ood",
    version="1.0.1",
    author="Shurui Gui",
    author_email="shurui.gui@tamu.edu",
    description="LECI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='GPLv3',
    url="https://github.com/divelab/LECI",
    project_urls={
        "Bug Tracker": "https://github.com/divelab/LECI/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    package_dir={"GOOD": "GOOD"},
    install_requires=install_requires,
    entry_points = {
        'console_scripts': [
            'goodtg = GOOD.kernel.main:goodtg',
            'goodtl = GOOD.kernel.launch:launch'
        ]
    },
    python_requires=">=3.8",
)
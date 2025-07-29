from setuptools import setup, find_packages

setup(
    name="irl_project",
    version="0.1",
    packages=find_packages(include=["irl", "envs", "models", "experiments", "tests"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "gymnasium",
        "torch",
        "seaborn",
        "tqdm",
        "PyYAML",
        "pandas",
        "networkx"
    ],
    entry_points={
        "console_scripts": []
    },
)

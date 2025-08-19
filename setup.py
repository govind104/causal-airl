from setuptools import setup, find_packages

setup(
    name="causal_irl",
    version="0.1",
    packages=find_packages(include=["irl", "envs", "models", "experiments", "tests", "visualisation"]),
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "matplotlib",
        "gymnasium[classic_control]",
        "seaborn",
        "pandas",
        "tqdm",
        "PyYAML",
        "networkx",
        "psutil",
        "scikit-learn"
    ],
    entry_points={
        "console_scripts": []
    },
)

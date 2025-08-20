from setuptools import setup, find_packages

setup(
    name="causal_irl",
    version="0.1",
    packages=find_packages(include=["irl", "envs", "models", "experiments", "tests", "visualisation"]),
    python_requires=">=3.10,<3.12",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "torch",
        "matplotlib",
        "PyYAML",
        "networkx",
        "psutil",
        "gymnasium[classic_control]",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    entry_points={
        "console_scripts": []
    },
)
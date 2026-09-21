from setuptools import find_packages, setup


setup(
    name="so101-policies",
    version="0.1.0",
    description=(
        "Scripted and learned policies for the SO-101 robot."
    ),
    python_requires=">=3.10",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "so101_policies": [
            "config/*.yaml",
        ],
    },
    install_requires=[
        "numpy>=1.23",
        "torch>=2.0",
    ],
    extras_require={
        "dataset": [
            "lerobot>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            (
                "so101-train-vla="
                "so101_policies.training.train:main"
            ),
            (
                "so101-vla-policy="
                "so101_policies.inference.policy_node:main"
            ),
        ],
    },
    zip_safe=False,
)
from setuptools import setup

package_name = "bringbackshapes"

setup(
    name=package_name,
    version="0.0.1",
    packages=[
        package_name,
        package_name + ".gym_wrappers",
        package_name + ".twod_playground",
    ],
    package_dir={"": "python"},
    license="BSD 3-Clause",
)

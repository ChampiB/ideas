from setuptools import find_packages
from setuptools import setup

# If this does not work, do pip install --upgrade setuptools==66 then pip install --upgrade wheel==0.38.4 and retry

setup(
    name="ideas",
    version="0.1.0",
    description="Library implementing various ideas",
    author="Theophile Champion",
    author_email="lb732@kent.ac.uk",
    url="https://git.cs.kent.ac.uk/mg483/metarl/",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    scripts=["scripts/run_task"],
    install_requires=[
        "gym",
        "torch==2.0.0",
        "numpy==1.23.4",
    ],
    python_requires="~=3.11",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="pytorch, machine learning, reinforcement learning, deep learning"
)

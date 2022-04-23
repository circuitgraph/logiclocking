import setuptools

with open("README.md") as fh:
    long_description = fh.read()

setuptools.setup(
    name="logiclocking",
    version="0.0.1",
    author="Ruben Purdy, Joseph Sweeney",
    author_email="rpurdy@andrew.cmu.edu, joesweeney@cmu.edu",
    description="Tools for locking circuits and attacking locked circuits.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/circuitgraph/logiclocking",
    project_urls={
        "Documentation": "https://circuitgraph.github.io/logiclocking/",
        "Source": "https://github.com/circuitgraph/logiclocking",
    },
    include_package_data=True,
    packages=["logiclocking"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["circuitgraph>=0.2.0"],
)

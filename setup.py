import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    include_package_data=True,
    name="dermoscopic_symmetry",
    version="0.0.1",
    author="Vincent Toureau",
    author_email="Vincent.Toureau@grenoble-inp.org",
    description="A package to study skin lesion's symmetry and help diagnose diseases like menalomas.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GalakVince/dermoscopic_symmetry",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

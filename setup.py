from setuptools import setup

# read readme as long description
with open("README.md", "r") as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="odmd",
    version="1.0.0",
    description="Online and Window Dynamic Mode Decomposition algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haozhg/odmd",
    author="Hao Zhang",
    author_email="haozhang@alumni.princeton.edu",
    # license="MIT",
    # classifiers=[
    #     "License :: OSI Approved :: MIT License",
    #     "Programming Language :: Python :: 3",
    #     "Programming Language :: Python :: 3.7",
    # ],
    packages=["odmd"],
    include_package_data=False,
    install_requires=["numpy"],
)

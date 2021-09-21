from setuptools import setup, Extension
from setuptools import find_packages
from os import listdir

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

scripts = ["scripts/"+i for i in listdir("scripts")]

if __name__ == "__main__":
    setup(
        name="suara-kami-community",
        version="1.0.1",
        scripts=scripts,
        description="Bahasa Malaysia Speech to Text",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Sani",
        author_email="khursani@omesti.com",
        url="https://github.com/redapesolutions/suara-kami-community",
        license="MIT License",
        packages=find_packages(),
        platforms=["linux", "unix","windows"],
        python_requires=">3.6",
        install_requires=required
    )
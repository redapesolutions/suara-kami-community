from setuptools import setup, Extension
from setuptools import find_packages

with open("README.md") as f:
    long_description = f.read()

if __name__ == "__main__":
    setup(
        name="suara-kami-community",
        version="0.1",
        description="Bahasa Malaysia Speech to Text",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Sani",
        author_email="khursani@omesti.com",
        url="https://github.com/redapesolutions/suara-kami-community",
        license="MIT License",
        packages=find_packages(),
        platforms=["linux", "unix"],
        python_requires=">3.6",
    )
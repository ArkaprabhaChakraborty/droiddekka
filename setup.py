from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = 'droiddekka',
    version = '0.0.0',
    author = "Arkaprabha Chakraborty",
    author_email="chakrabortyarkaprabha998@gmail.com",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/ArkaprabhaChakraborty/droidekka",
    packages = find_packages(exclude=["contrib"]),
    install_requires=[
        'numpy', 'scipy', 'matplotlib'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],

    
    )
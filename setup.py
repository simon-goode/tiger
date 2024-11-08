import setuptools

LONG_DESC = "<LONG DESC>"

PROJECT_URLS = {"Source": "https://github.com/simon-goode/tiger"}
INSTALL_REQUIRES = ["pandas>=0.24", "jinja2",
                    "scipy", "numpy>=1.16",
                    "statsmodels>=0.9", "patsy",
                    "seaborn>=0.9", "matplotlib>=3"]

setuptools.setup(
    name="tiger",
    version="0.1.0",
    author="Simon Goode",
    description="Econometrics Library for Python",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    keywords=["econometrics", "regression", "statistics", "economics", "models"],
    url="https://github.com/simon-goode/tiger",
    project_urls=PROJECT_URLS,
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    license="MIT"
)
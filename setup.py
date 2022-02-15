import re
import setuptools

(__version__,) = re.findall("__version__.*\s*=\s*[']([^']+)[']",
                            open('run_xgb/__init__.py').read())

setuptools.setup(
    name="run_xgb",
    version=__version__,
    packages=setuptools.find_packages(),
    python_requires="<3.9.0",
    install_requires=[
        "matplotlib==3.5.1",
        "numpy==1.21.5",
        "pandas==1.3.5",
        "pyyaml==6.0",
        "pytest==6.2.5",
        "scikit-learn==1.0.2",
        "xgboost==1.5.2"
    ],
)
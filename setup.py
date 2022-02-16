from setuptools import setup, find_packages

setup(
    name='dl_portfolio',
    version='1.1.1',
    url='https://github.com/BrunoSpilak/dl-portfolio.git',
    author='Bruno Spilak',
    author_email='bruno.spilak@gmail.com',
    dependency_links=[],
    python_requires='~=3.8',
    install_requires=[
        "numpy>=1.19.2",
        "pandas==1.1.5",
        "scipy==1.6.0",
        "tensorflow==2.4.0",
        "tensorflow-probability==0.12.2",
        "matplotlib==3.2.2",
        "seaborn==0.11.1",
        "pyportfolioopt==1.4.1",
        "riskparityportfolio==0.2",
        "joblib==1.0.0",
        "scikit-learn==0.24.0",
        "fastcluster==1.2.3",
        "pytest==6.2.5",
        # "jaxlib-0.3.0"
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    zip_safe=False,
    packages=find_packages()
)

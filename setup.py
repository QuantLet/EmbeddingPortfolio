from setuptools import setup, find_packages

setup(
    name='dl_portfolio',
    version='0.0.0',
    url='https://github.com/BrunoSpilak/dl-portfolio.git',
    author='Bruno Spilak',
    author_email='bruno.spilak@gmail.com',
    dependency_links=[],
    python_requires='~=3.8',
    install_requires=[
        "numpy>=1.19.2",
        "pandas==1.1.5",
        "scipy==1.6.0",
        "riskfolio-lib",
        "tensorflow==2.4.0",
        "tensorflow-probability==0.12.2",
        "jax[cpu]",
        "matplotlib>=3.2.2",
        "seaborn==0.11.1",
        "pyportfolioopt==1.4.1",
        "riskparityportfolio==0.2",
        "joblib>=1.0.0",
        "scikit-learn>=0.24.0",
        "fastcluster==1.2.6",
        "statsmodels==0.13.5",
        "arch>=5.3.1",
        "gap-stat==2.0.2"
    ],
    zip_safe=False,
    packages=find_packages()
)

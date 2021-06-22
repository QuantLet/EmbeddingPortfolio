from setuptools import setup, find_packages

setup(
    name='dl_portfolio',
    version='0.1.1',
    url='https://github.com/BrunoSpilak/dl-portfolio.git',
    author='Bruno Spilak',
    author_email='bruno.spilak@gmail.com',
    dependency_links=[],
    install_requires=[
        "tensorflow==2.4.0",
        "numpy==1.19.4",
        "pandas==1.1.4",
        "scipy==1.5.4",
        "matplotlib==3.3.3",
        "seaborn==0.11.1",
        "pyportfolioopt==1.4.1"
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    zip_safe=False,
    packages=find_packages()
)

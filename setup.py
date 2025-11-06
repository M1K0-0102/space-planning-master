from setuptools import setup, find_namespace_packages

setup(
    name="interior-design-pipeline",
    version="0.1",
    packages=find_namespace_packages(include=['src.*']),
    package_dir={'': '.'},
    install_requires=[
        'numpy',
        'torch',
        'opencv-python',
        'scikit-learn',
        'pytest',
        'pytest-cov',
        'pytest-mock'
    ]
) 
from setuptools import setup, find_namespace_packages

setup(
    name='simulator_aspire',
    version='0.0.1',
    author='',
    author_email='',
    license='MIT',
    python_requires='>=3.10',
    description="",
    url="",
    long_description=open('readme.md').read(),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(exclude=['tests','tests.*']),
    install_requires=['rdkit', 'rdflib', 'twa>=0.0.4a0'],
    include_package_data=True
)

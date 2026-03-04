from setuptools import setup,find_packages

setup(
    name='pycod',
    version='1.0',
    description='Python codes to compute and analyse Complex Orthogonal Decomposition (COD) on multidimensional signals',
#      url='needs a URL',
    author='Marc Vacher, Stéphane Perrard',
    author_email='stephane.perrard@espci.fr',
    license='GNU',
    packages=find_packages(),
    zip_safe=False,
#      package_data={'tangle': ['cl_src/*.cl']})
)

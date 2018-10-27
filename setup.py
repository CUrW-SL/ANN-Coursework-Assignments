from setuptools import setup, find_packages

setup(
    name='ANN',
    version='0.0.1',
    packages=find_packages(),
    license='MIT',
    author='thilinamad',
    author_email='madumalt@gamil.com',
    description='Project for ANN module',
    install_requires=['pandas', 'scikit-learn', 'keras', 'numpy'],
    zip_safe=True
)

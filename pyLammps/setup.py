from setuptools import setup

setup(
    name='pyLammps',
    version='0.1.1',
    author='Antonio Tripodo and Gianfranco Cordella',
    author_email='a.tripodo90@gmail.com',
    install_requires=['numpy', 'scipy','pandas','freud-analysis'],
    packages=['pyLammps'],
)

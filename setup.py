from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT='_e .'
def get_requirements(file_path:str)-> List[str]:
    '''
    this function returns list of modules needed to run the package overall
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        print(requirements)

        requirements=[req.replace('\n',"") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)


        print(requirements)
    return requirements

from setuptools import setup, find_packages

setup(
    name="MLTutorial",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "requests",
        "seaborn",
        "matplotlib",
        "scikit-learn",
        "catboost",
        "xgboost",
        "dill"
    ],
)
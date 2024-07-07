from setuptools import setup

setup(
    name="gym_SelfHealing",
    version="0.0.1",
    install_requires=["gymnasium==0.28.1", 
                      "numpy",
                      "pandas",
                      "matplotlib",
                      "stable_baselines3==2.0.0",
                      "torch==2.0.1"
                      "juliacall"                   
                        ],
)
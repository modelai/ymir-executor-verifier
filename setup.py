from setuptools import find_packages, setup

__version__ = '0.0.2'

requirements = []
with open('requirements.txt') as f:
    for line in f.read().splitlines():
        requirements.append(line)

setup(name='ymir-verifier',
      version=__version__,
      python_requires=">=3.7",
      install_requires=requirements,
      author_email="wang.jiaxin@intellif.com",
      description="ymir executor check tools, auto test the ymir docker image for training, mining and infer",
      url='https://github.com/modelai/ymir-executor-verifier',
      packages=find_packages(exclude=["*tests*"]),
      include_package_data=True,
      entry_points={'console_scripts': [
          'ymir-verifier = src.cmd:main',
      ]})

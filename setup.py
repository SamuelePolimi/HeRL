from setuptools import setup

setup(name='herl',
      version='0.1.0',
      description='Library for Reinforcement Learning tools.',
      author='Intelligent Autonomous Systems Lab',
      author_email='samuele.tosatto@tu-darmstadt.de',
      license='MIT',
      packages=['herl'],
      zip_safe=False,
      install_requires=[
          'numpy>=1.17.3',
          'gym>=0.15.4'
      ])

from setuptools import setup

setup(name='chainer-computational-cost',
      version='0.1',
      description='Chainer theoretical computational cost estimator',
      author='Daichi SUZUO',
      author_email='suzuo@preferred.jp',
      packages=[
          'chainer_computational_cost'
      ],
      test_require=[
          'pytest>=3.0.0'
      ],
      install_requires=[
          'chainer>=4.0.0',
          'texttable>=1.2.0'
      ],
      license='MIT')

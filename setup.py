from setuptools import setup

setup(name='chainer-computational-cost',
      version='0.1',
      description='Chainer theoretical computational cost estimator',
      author='Daichi SUZUO',
      author_email='suzuo@preferred.jp',
      packages=[
          'chainer_computational_cost'
      ],
      install_requires=[
          'chainer>=4.0.0'
      ],
      license='MIT'
)


from setuptools import setup, find_packages

setup(
    name='VecKM',
    version='1.0.0',
    packages=find_packages(),
    author='Dehao Yuan',
    author_email='dhyuan@umd.edu',
    description='A very efficient and descriptive local geometry encoder / point tokenizer / patch embedder.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dhyuan99/VecKM/',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)



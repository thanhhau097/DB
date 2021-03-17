from setuptools import setup
from io import open
from setuptools import find_packages


with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()
    requirements.append('tqdm')


def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README


setup(
    name='lioneldb',
    packages=find_packages(),
    #package_dir={'lionelocr': ''},
    # package_data={'': ['lioneldb/assets/*']},
    include_package_data=True,
    # entry_points={"console_scripts": ["paddleocr= paddleocr.paddleocr:main"]},
    version='0.0.1',
    install_requires=requirements,
    license='Apache License 2.0',
    description='Lionel Text Detection',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/thanhhau097/DB',
    download_url='https://github.com/thanhhau097/DB.git',
    keywords=[
        'ocr'
    ],
    classifiers=[
        'Intended Audience :: Developers', 'Operating System :: OS Independent',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7', 'Topic :: Utilities'
    ], )
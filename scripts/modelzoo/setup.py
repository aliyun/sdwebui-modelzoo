import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='modelzoo',
    version='0.1',
    author='Zhoulou.wzh, Hongzhua.ylz',
    author_email='zhoulou.wzh@alibaba-inc.com',
    description='Modelzoo Management with ModelScope-gitlab ' +
    'lfs and DocArray multimedia search extension',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://code.alibaba-inc.com/' +
    'zhoulou.wzh/modelzoo/blob/master/README.md',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'pai-docarray',
        'modelscope',
    ],
    python_requires='>=3.7',
)

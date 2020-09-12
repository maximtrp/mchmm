from setuptools import setup
from os.path import join, dirname

setup(
    name='mchmm',
    version='0.4.0',
    description='Markov chains and Hidden Markov models',
    long_description=open(join(dirname(__file__), 'DESCRIPTION.rst')).read(),
    url='http://github.com/maximtrp/mchmm',
    author='Maksim Terpilowski',
    author_email='maximtrp@gmail.com',
    license='BSD',
    packages=['mchmm'],
    keywords='markov chain hidden markov models',
    install_requires=['numpy', 'scipy', 'graphviz'],
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',

        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    test_suite='tests.mchmm_suite',
    zip_safe=False
)

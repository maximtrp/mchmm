from setuptools import setup
from os.path import join, dirname

setup(name='mchmm',
      version='0.1.0',
      description='Markov chains and Hidden Markov models',
      long_description=open(join(dirname(__file__), 'DESCRIPTION.rst')).read(),
      url='http://github.com/maximtrp/mchmm',
      author='Maksim Terpilowski',
      author_email='maximtrp@gmail.com',
      license='BSD',
      packages=['mchmm'],
      keywords='markovchain hmm',
      install_requires=['numpy', 'scipy'],
	  classifiers=[
		'Development Status :: 3 - Alpha',

		'Intended Audience :: Education',
		'Intended Audience :: Information Technology',
		'Intended Audience :: Science/Research',

		'Topic :: Scientific/Engineering :: Information Analysis',
		'Topic :: Scientific/Engineering :: Mathematics',

		'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.1',
		'Programming Language :: Python :: 3.2',
		'Programming Language :: Python :: 3.3',
		'Programming Language :: Python :: 3.4',
		'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
	  ],
      test_suite='tests.mchmm_suite',
      zip_safe=False)

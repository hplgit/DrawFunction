from distutils.core import setup

# Installation of just one module (no package)
setup(name='DrawFunction',
      version='0.1',
      author'Hans Petter Langtangen <hpl@simula.no>',
      package_dir={'': 'src'},     # needed if module is not in current dir
      py_modules=['DrawFunction'],
      )

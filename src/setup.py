from distutils.core import setup, Extension
import os
import numpy

os.environ["CC"] = "g++ -Ofast -lconfig++ -lgsl"
lif = Extension("stdp", ["simlifmodule.cpp"],
                include_dirs=[
                    '/usr/local/include', '/usr/local/include/c++',
                    numpy.get_include()
                ],
                extra_link_args=['-lconfig++'],
                language="c++")
setup(name="PackageName", ext_modules=[lif])

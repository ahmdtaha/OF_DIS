from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from glob import glob
import numpy
import sysconfig

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-std=c++11", "-Wall", "-Wextra", "-D SELECTMODE=1","-D SELECTCHANNEL=3"]

sourcefiles = ['pyx_flow.pyx',"run_dense.cpp" ,"oflow.cpp", "patch.cpp", "patchgrid.cpp", "refine_variational.cpp" ,"FDF1.0.1/image.c", "FDF1.0.1/opticalflow_aux.c", "FDF1.0.1/solver.c"]
#sourcefiles.append("msrc.cpp")
#sourcefiles = ['pyx_flow.pyx',"run_dense.cpp"]
#print(sourcefiles)
extensions = [Extension("pyx_flow", sourcefiles , include_dirs=[numpy.get_include()],
                    extra_compile_args=extra_compile_args,
                    #library_dirs=['/opt/local/stow/opencv-3.2.0/lib'],
                    library_dirs=['/usr/local/lib'],
                    #include_dirs=[numpy.get_include(),'/opt/local/stow/opencv-3.2.0/include'],
                    libraries=['opencv_core','opencv_highgui','opencv_imgproc','opencv_imgcodecs'])]
setup(
  name = "pyxtutorial",
  #ext_modules=[ Extension("cmainmodule", ["cmainmodule.pyx", "main.cpp"]) ]
  # ext_modules=[
  #         Extension("pyx_flow", sourcefiles , include_dirs=[numpy.get_include()],
  #             library_dirs = [os.getcwd(),],  # path to .a or .so file(s)
  #             extra_compile_args=extra_compile_args,
  #             language='c++11',
  #         ),
  #     ]
  ext_modules=cythonize(extensions),
  include_dirs=[numpy.get_include(),'/usr/local/include/opencv2','/usr/local/include/eigen3']
)
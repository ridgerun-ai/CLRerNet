from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='nms',
    packages=['nms'],
    package_dir={'': 'src'},
    ext_modules=[
        CppExtension(
            name='nms.details',
            sources=['src/nms.cpp']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)

import sys
import subprocess
from setuptools import setup

MINIMUM_SUPPORTED_PYTHON_VERSION = "3.7"

class MinPythonVersion(distutils.cmd.Command):
    description = "Print out the minimum supported Python version"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print(MINIMUM_SUPPORTED_PYTHON_VERSION)

if sys.argv[-1] == 'setup.py':
    print('To install, run \'python setup.py install\'')
    print()

if sys.version_info[:2] < (3, 5):
    print(('lingatagger requires Python version 3.5 or later (%d.%d detected).' %sys.version_info[:2]))
    sys.exit(-1)


if __name__ == "__main__":
    setup(
        name = "newron",
        version = "0.1",
        author = "Newron AI",
        author_email = "help@newron.ai",
        description = "Supercharge MLFlow with Newron.ai",
        url='https://github.com/NewronAI/newron-mlflow-client',
        keywords= ['mlops', "experiment", "tracking", "deployments", "mlflow"],
        packages = ['newron'],
        license = 'Apache License',
        install_requires = ["mlflow"],
        entry_points="""
                        [console_scripts]
                        newron=newron.cli:cli
                    """
        cmdclass={
        "min_python_version": MinPythonVersion,
                 },
    )

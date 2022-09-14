import sys
import subprocess
from setuptools import setup
import distutils

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


if __name__ == "__main__":
    setup(
        name = "newron-sdk",
        version = "0.1.2",
        author = "Newron AI",
        author_email = "hello@newron.ai",
        description = "NewronAI: Machine Learning, Made Simple. Client SDK for Newron AI",
        url='https://newron.ai',
        project_urls={
                'Documentation': 'https://docs.newron.ai/',
                'Source': 'https://github.com/NewronAI/NewronSDK/',
                'Tracker': 'https://github.com/NewronAI/NewronSDK/issues',
            },
        keywords= ['mlops', "experiment", "tracking", "deployments", "mlflow"],
        packages = ['newron'],
        license = 'Apache License',
        install_requires = ["mlflow"],
        entry_points="""
                        [console_scripts]
                        newron=newron.cli:cli
                    """,
        cmdclass={
        "min_python_version": MinPythonVersion,
                 },
    )

from setuptools import setup, find_packages

setup(
    name="newron-plugin",
    version="0.0.1",
    description="Newron Plugin for MLFlow",
    packages=find_packages(),
    # Require MLflow as a dependency of the plugin, so that plugin users can simply install
    # the plugin & then immediately use it with MLflow
    install_requires=["mlflow"],
    entry_points={
        "mlflow.request_header_provider": "unused=plugin.request_header_provider:PluginRequestHeaderProvider",
        # pylint: disable=line-too-long
    },
)

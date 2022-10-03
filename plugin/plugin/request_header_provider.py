import os

from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider


class PluginRequestHeaderProvider(RequestHeaderProvider):
    """RequestHeaderProvider provided through plugin system"""

    def in_context(self):
        return False

    def request_headers(self):
        headers = {}
        if os.environ.get("NEWRON_ACCESS_TOKEN") is not None:
            headers["Authorization"] = "Bearer {}".format(os.environ.get("NEWRON_ACCESS_TOKEN"))
        headers["X-MLflow-User"] = "newron"
        headers["test"] = "test"
        print("adding headers")
        return headers

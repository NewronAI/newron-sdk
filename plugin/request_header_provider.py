import os

from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider

in_context = True

class PluginRequestHeaderProvider(RequestHeaderProvider):
    """RequestHeaderProvider provided through plugin system"""

    def in_context(self):
        # print("in context", in_context)
        return in_context

    def request_headers(self):
        headers = {}
        print(os.environ.get("NEWRON_ACCESS_TOKEN"))
        if os.environ.get("NEWRON_ACCESS_TOKEN") is not None:
            headers["Authorization"] = "Bearer {}".format(os.environ.get("NEWRON_ACCESS_TOKEN"))
        headers["X-MLflow-User"] = "newron"
        headers["test"] = "test"
        # print("adding headers", headers)
        return headers


if __name__ == "__main__":
    print("running plugin")
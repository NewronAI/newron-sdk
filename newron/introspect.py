import time
import requests

from newron.colored import warning_print, error_print, blue_print, success_print
from newron.token_store import TokenStore
from newron.version import __version__

introspection_validity = 24 * 60 * 60  # 24 hours


def get_newron_version():
    return __version__


def get_newron_version_from_pypi():
    r = requests.get("https://pypi.org/pypi/newron/json")
    return r.json()["info"]["version"]


def get_newron_version_from_github():
    version = None
    try:
        r = requests.get("https://api.github.com/repos/NewronAI/newron-sdk/releases/latest")
        version = r.json()["tag_name"]
    except Exception as e:
        pass
    return version


def introspect():
    token_store = TokenStore()
    last_introspection = token_store.get_introspected_at()

    if last_introspection is not None and time.time() - last_introspection < introspection_validity:
        # cyan_print("Last version check was less than 24 hours ago. Skipping...")
        return

    success_print("{}".format(get_newron_version_from_pypi()))
    print("Newron Version from Github: {}".format(get_newron_version_from_github()))

    try:
        r = requests.get("https://api.newron.ai/v1/introspect/sdk-version", params={"version": get_newron_version()})
        data = r.json()
        status = data["status"]
        comment = data["comment"]
        messages = data["messages"]

        if status == "DEPRECATED":
            warning_print("Current Installed Newron SDK is deprecated. Please upgrade to the latest version.")
        elif status == "INACTIVE":
            error_print("Current Installed Newron SDK is Outdated. Please upgrade to the latest version.")

        blue_print(comment)

        for message in messages:
            if message["type"] == "INFO":
                blue_print("[INFO] "+message["message"])
            elif message["type"] == "WARNING":
                warning_print("[WARN] " + message["message"])
            elif message["type"] == "ERROR":
                error_print("[ERR] " + message["message"])

        token_store.set_introspected_at(time.time())

    except Exception as e:
        pass


if __name__ == "__main__":
    introspect()


# Copyright 2018 Databricks, Inc.
import re


VERSION = "0.01.dev"


def is_release_version():
    return bool(re.match(r"^\d+\.\d+$", VERSION))
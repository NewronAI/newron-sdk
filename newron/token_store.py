import time
import pickle
import logging
import os
from pathlib import Path

from newron.config_path_utils import find_config_file, get_path_in_home_dir

logging.basicConfig(format='[TOKEN_STORE] %(message)s', level=logging.DEBUG)

_CONFIG_FILENAME = "runtime.config.pkl"


class TokenStore:
    def __init__(self):
        logging.info("Initializing token store")
        logging.info("Checking token in current and parent directories")
        is_config_created = True
        self._config_file_path = find_config_file(_CONFIG_FILENAME)
        # If config file is not found in current and parent directories, create it in home directory
        # Bug: if .newron exists in root and config file is not found create in that parent itself

        if self._config_file_path is None:
            logging.info("Checking token in home directory")
            self._config_file_path = find_config_file(_CONFIG_FILENAME, Path().home())

        if self._config_file_path is None:
            self._config_file_path = get_path_in_home_dir(file_name=_CONFIG_FILENAME)
            is_config_created = False

        try:
            logging.info("Loading token store from file")
            self._db_file = open(self._config_file_path, "rb")
            db = pickle.load(self._db_file)

            self._refreshToken = None
            self._authToken = None
            self._expiresAt = None

            if "refresh_token" in db:
                self._refreshToken = db["refresh_token"]

            if "auth_token" in db:
                self._authToken = db["auth_token"]

            if "expires_at" in db:
                self._expiresAt = db["expires_at"]

        except FileNotFoundError as e:
            logging.warning("Token store file not found")
            self._refreshToken = None
            self._authToken = None
            self._expiresAt = None

            if os.environ.get("REFRESH_TOKEN") is not None:
                logging.info("Setting refresh token from environment variable")
                self._refreshToken = os.environ.get("REFRESH_TOKEN")
                self.save()


    def get_refresh_token(self) -> str:
        logging.info("Getting refresh token")
        return self._refreshToken

    def get_auth_token(self) -> str:
        logging.info("Getting auth token")
        return self._authToken

    def get_expires_at(self) -> int:
        logging.info("Getting expires at")
        return self._expiresAt

    def set_refresh_token(self, refresh_token) -> None:
        logging.info("Setting refresh token")
        self._refreshToken = refresh_token
        self.save()

    def set_auth_token(self, auth_token) -> None:
        logging.info("Setting auth token")
        self._authToken = auth_token
        self.save()

    def set_expires_at(self, expires_at) -> None:
        logging.info("Setting expires at")
        self._expiresAt = expires_at
        self.save()

    def is_expired(self) -> bool:
        logging.info("Checking if token is expired")
        return self._expiresAt < time.time()

    def is_valid(self) -> bool:
        logging.info("Checking if token store is valid")
        return self._refreshToken is not None and self._authToken is not None

    def save(self):
        logging.info("Saving token store to file")
        try:
            logging.info("Opening token store file")
            db_file = open(self._config_file_path, "wb")
            db = {}
            if self._refreshToken is not None:
                logging.info("Saving refresh token")
                db["refresh_token"] = self._refreshToken
            if self._authToken is not None:
                logging.info("Saving auth token")
                db["auth_token"] = self._authToken
            if self._expiresAt is not None:
                logging.info("Saving expires at")
                db["expires_at"] = self._expiresAt
            logging.info("Saving token store to file")
            pickle.dump(db, db_file)
            logging.info("Flushing token store file")
            db_file.flush()
            logging.info("Closing token store file")
            db_file.close()
            logging.info("Token store saved")
        except Exception as e:
            logging.error("Failed to save token store to file")


    def __del__(self):
        logging.info("Destroying token store")
        try:
            self._db_file.close()
        except Exception as e:
            pass

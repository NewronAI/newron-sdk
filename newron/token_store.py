import time
import pickle
import logging
import os
from pathlib import Path

from newron.config_path_utils import find_config_file, get_path_in_home_dir, find_config_folder

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

_CONFIG_FILENAME = "runtime.config.pkl"


class TokenStore:
    def __init__(self):
        logger.info("Initializing token store")

        is_config_created = True

        self._config_file_path = None

        # Beautifully chained if statements upcoming
        logger.info("Checking token in current and parent directories")
        self._config_file_path = find_config_file(_CONFIG_FILENAME)

        # If config file is not found in current and parent directories, create it in home directory
        if self._config_file_path is None:
            logger.info("Checking token in home directory")
            self._config_file_path = find_config_file(_CONFIG_FILENAME, Path().home())

        if self._config_file_path is None:
            is_config_created = False
            logger.info("Looking for .newron folder in home directory and its parent directories")
            config_folder = find_config_folder()

            if config_folder is not None:
                self._config_file_path = get_path_in_home_dir(file_name=_CONFIG_FILENAME, path=config_folder)

        if self._config_file_path is None:
            try:
                os.makedirs(get_path_in_home_dir(path=".newron"))
            except FileExistsError:
                pass
            except Exception as e:
                logger.error("Failed to create .newron folder in home directory. Seems like you don't have permissions")
                raise e
            self._config_file_path = get_path_in_home_dir(file_name=_CONFIG_FILENAME)

        try:
            if not is_config_created:
                raise FileNotFoundError("Config file does not exists")

            logger.info("Loading token store from file")
            self._db_file = open(self._config_file_path, "rb")
            db = pickle.load(self._db_file)

            self._refreshToken = None
            self._authToken = None
            self._expiresAt = None
            self._introspected_at = None

            if "refresh_token" in db:
                self._refreshToken = db["refresh_token"]

            if "auth_token" in db:
                self._authToken = db["auth_token"]

            if "expires_at" in db:
                self._expiresAt = db["expires_at"]

            if "introspected_at" in db:
                self._introspected_at = db["introspected_at"]

        except FileNotFoundError as e:
            logger.warning("Token store file not found")
            self._refreshToken = None
            self._authToken = None
            self._expiresAt = None
            self._introspected_at = None

            if os.environ.get("NEWRON_ACCESS_TOKEN") is not None:
                logger.info("Setting refresh token from environment variable")
                self._refreshToken = os.environ.get("NEWRON_ACCESS_TOKEN")
                self.save()

    def get_refresh_token(self) -> str:
        logger.info("Getting refresh token")
        return self._refreshToken

    def get_auth_token(self) -> str:
        logger.info("Getting auth token")
        return self._authToken

    def get_expires_at(self) -> int:
        logger.info("Getting expires at")
        return self._expiresAt

    def get_introspected_at(self) -> int:
        return self._introspected_at

    def set_refresh_token(self, refresh_token) -> None:
        logger.info("Setting refresh token")
        self._refreshToken = refresh_token
        self.save()

    def set_auth_token(self, auth_token) -> None:
        logger.info("Setting auth token")
        self._authToken = auth_token
        self.save()

    def set_expires_at(self, expires_at) -> None:
        logger.info("Setting expires at")
        self._expiresAt = expires_at
        self.save()

    def set_introspected_at(self, introspected_at) -> None:
        logger.info("Setting introspected at")
        self._introspected_at = introspected_at
        self.save()

    def is_expired(self) -> bool:
        logger.info("Checking if token is expired")
        return self._expiresAt < time.time()

    def is_valid(self) -> bool:
        logger.info("Checking if token store is valid")
        return self._refreshToken is not None and self._authToken is not None

    def save(self):
        logger.info("Saving token store to file")
        try:
            logger.info("Opening token store file")

            db_file = open(self._config_file_path, "wb")
            db = {}

            if self._refreshToken is not None:
                logger.info("Saving refresh token")
                db["refresh_token"] = self._refreshToken

            if self._authToken is not None:
                logger.info("Saving auth token")
                db["auth_token"] = self._authToken

            if self._expiresAt is not None:
                logger.info("Saving expires at")
                db["expires_at"] = self._expiresAt

            if self._introspected_at is not None:
                logger.info("Saving introspected at")
                db["introspected_at"] = self._introspected_at

            logger.info("Saving token store to file")
            pickle.dump(db, db_file)

            logger.info("Flushing token store file")
            db_file.flush()

            logger.info("Closing token store file")
            db_file.close()

            logger.info("Token store saved")
        except Exception as e:
            logger.error("Failed to save token store to file")


    def __del__(self):
        logger.info("Destroying token store")
        try:
            self._db_file.close()
        except Exception as e:
            pass


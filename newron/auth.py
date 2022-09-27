import os
import requests
import time
import webbrowser
from sys import exit

from newron.token_store import TokenStore


class Auth0():
    token_store = TokenStore()

    auth0url = "https://auth.newron.ai"

    url = auth0url + '/oauth/device/code'
    userURL = auth0url + '/userinfo'
    clientId = "qhgH8CCF8riL9XYOZr8EPym8lxq3XEd3"

    audience = "https://api.newron.ai"
    clientCredentials = {"client_id": clientId, "scope": "openid email profile newron-server", "audience": audience}

    def __init__(self) -> None:
        pass

    def do_device_login(self):
        x = requests.post(self.url, json=self.clientCredentials)
        if x.status_code != 200:
            print("Error in verification: " + str(x.status_code))
            print(x.text)
            exit()
        resp = x.json()
        print("\n\n************************************************************\n\n")
        print("Please verify the following code on the device: " + str(resp['user_code']))
        print("\n\n************************************************************\n\n")
        print("Please visit the following link " + str(resp['verification_uri_complete']))
        print("\n\n************************************************************\n\n")
        # Step 2: Open Webbrowser
        webbrowser.open(resp["verification_uri_complete"])
        # Step 3: Poll for token
        poll_url = self.auth0url + "/oauth/token"
        poll_obj = {
            "client_id": self.clientId,
            "device_code": resp["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }
        max_polls = 60
        op = requests.post(poll_url, json=poll_obj)
        op_json = op.json()
        failed_polls = 0
        # print(op.text)
        # print(op.status_code)
        while op.status_code != 200 and max_polls > 0:
            op = requests.post(poll_url, json=poll_obj)
            op_json = op.json()
            max_polls -= 1
            if op.status_code == 200:
                print("Authorization Successful")
                break;
            if op.status_code == 400 or op.status_code == 401:
                print("Authorization Failed")
                exit();
            if op_json["error"] == "invalid_grant":
                print("Authorization Failed")
                break;
            if op_json["error"] == "authorization_pending":
                print("Authorization Pending")
                time.sleep(2)
            elif op_json["error"] == "slow_down":
                if failed_polls == 0:
                    time.sleep(10)
                else:
                    print("Waiting for Authorization")
                    time.sleep(8)
                failed_polls += 1
            else:
                time.sleep(10)
                break
        if op.status_code != 200 or max_polls == 0:
            # print("Error: " + op.text)
            exit()
        return op_json

    def authenticate(self):

        if self.token_store.is_valid() is False or self.token_store.is_expired() is True:
            auth_output = self.do_device_login()
            self.token_store.set_refresh_token(auth_output["id_token"])
            self.token_store.set_auth_token(auth_output["access_token"])
            self.token_store.set_expires_at(auth_output["expires_in"] + int(time.time()))

        token = self.token_store.get_auth_token()

        headers = {"Authorization": "Bearer " + token}
        user_response = requests.get(self.userURL, headers=headers)
        if not user_response.json()["email_verified"]:
            print("Please Verify your email")
        user_response = user_response.json()
        user_response["access_token"] = token
        return user_response


if __name__ == "__main__":
    auth = Auth0()
    print(auth.authenticate())

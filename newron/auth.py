import os
import requests
import time
import webbrowser

class Auth0():
    auth0url = "https://dev-pg1h84k1.us.auth0.com"
    url = auth0url + '/oauth/device/code'
    userURL = auth0url + '/userinfo'
    clientId = "qhgH8CCF8riL9XYOZr8EPym8lxq3XEd3"
    audience = "https://grpc-api-gateway-d8q71ttn.uc.gateway.dev/"
    clientCredentials = {"client_id": clientId , "scope" : "openid email profile newron-server" , "audience" : audience}

    def __init__(self) -> None:
        pass

    def authenticate(self):
        x = requests.post(self.url, json = self.clientCredentials)
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
        pollUrl = self.auth0url + "/oauth/token"
        pollObj = { 
            "client_id": self.clientId,
            "device_code": resp["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }
        maxPolls = 60
        op = requests.post(pollUrl, json = pollObj)
        opJson = op.json()
        failed_polls = 0
        # print(op.text)
        # print(op.status_code)
        while op.status_code != 200 and maxPolls > 0:
            op = requests.post(pollUrl, json = pollObj)
            opJson = op.json()
            maxPolls -= 1
            if op.status_code == 200:
                print("Authorization Successful")
                break;
            if op.status_code == 400 or op.status_code == 401:
                print("Authorization Failed")
                exit();
            if opJson["error"] == "invalid_grant":
                print("Authorization Failed")
                break;
            if opJson["error"] == "authorization_pending":
                print("Authorization Pending")
                time.sleep(2)
            elif opJson["error"] == "slow_down":
                if failed_polls == 0:
                    time.sleep(10)
                else:
                    print("Waiting for Authorization")
                    time.sleep(8)
                failed_polls+=1
            else:
                time.sleep(10)
                break
        if(op.status_code != 200 or maxPolls == 0):
            # print("Error: " + op.text)
            exit()
        token = op.json()["access_token"]
        headers = {"Authorization": "Bearer " + token}
        userResponse = requests.get(self.userURL, headers = headers)
        if not userResponse.json()["email_verified"]:
            print("Pease Verify your email")
        userResponse = userResponse.json()
        userResponse["access_token"] = token
        return userResponse
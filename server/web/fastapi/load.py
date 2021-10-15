import time
import json
from locust import HttpUser, task, between
class QuickstartUser(HttpUser):
    wait_time = between(1, 3)
    @task(1)
    def testFlask(self):
        data = {'name': 'name', 'label': 'label'}
        files = {'file': open('test.wav', 'rb')}
        self.client.post("/transcript", data=data, files=files)
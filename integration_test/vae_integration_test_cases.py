import unittest

import subprocess
import unittest
import time
import requests


class MyTestCase(unittest.TestCase):
    SERVER_URL = "http://localhost:8000"
    UP_URL = f"{SERVER_URL}/health"
    server_process = None

    @classmethod
    def setUpClass(cls):
        #   python -m vae.api
        cls.server_process = subprocess.Popen(
            ["poetry", "run", "python", "-m", "vae.api", "--host", "127.0.0.1", "--port", "8000"]
        )
        timeout = 10
        start_time = time.time()
        while True:
            try:
                r = requests.get(cls.UP_URL)
                print(r.json())
                if r.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                pass
            if time.time() - start_time > timeout:
                raise TimeoutError("Server did not start in time")
            time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait()

    def test_server_starts(self):
        response = requests.get(self.UP_URL)
        self.assertEqual(response.status_code, 200)

    def test_health_endpoing_response(self):
        response = requests.get(self.UP_URL)
        data = response.json()
        self.assertEqual(data.get("status"), "running")
        self.assertTrue(data.get("model_loaded"))
        self.assertIn("timestamp", data)
        self.assertIn("uptime_seconds", data)


if __name__ == '__main__':
    unittest.main()

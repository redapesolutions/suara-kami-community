#! /usr/bin/env python
### feedback
from smart_open import open
from google.cloud.storage import Client
from google.oauth2 import service_account
import uuid
import json
from datetime import datetime
from pathlib import Path
from smart_open.smart_open_lib import patch_pathlib
patch_pathlib()
import shutil

info_path = Path.home()/".sk/info.txt"

def upload(path):
    path = Path(path)
    config = {
        "type": "service_account",
        "project_id": "suara-kami-271607",
        "private_key_id": "9b59c8caebf2482636d49c0f0d2724bb194ddb1d",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC1I2autoJU7lC2\n2KDs0YipMd4nuvUuur7r6QYojbf3bjSQeCcjZIL1MHnxE6/N4Psm7udOV/eU7Y0R\naW9o4aYDWSZp7NzVErOvh9Ezlpvaw9kyk+XDmopFwuZi3Y2THfeO957MfcIuiqqA\niDZ+T73lZdT+esTXeYmMSXTyxGW/1Ot3nlzTYtRdvg+vjLIjltfliOYMf0nEAccF\ns47l2W9d18apj6tqNftXgL4Mz+xaGuY4FLDisIAQkpR9TL+cUfTQOxjgoXUeBEHQ\nONvxIfN51OsOJE2OuAMj8fpLl8K7AUgVai0bk0Aj+YrbaHIegNTRNdivoqEfSrSN\n57GmYfZlAgMBAAECggEAAYXLjxKdenPjlzRJK54NSiLPRvU4d2aX3g+Zsib5ODa9\nPi0/lizga70D4WXORGqsocsM6oJgMRvLSkcUUF58andjrizdpFrhlzGklAyJjdq3\nzWGJ4zPxQp0QGh4NQRw8E8yud7784WHgBTzGSUruG54Lns7OyPyBp/2I4qHwNJ8X\n1ILb7ClVP7PB7QHeQOZ5u/yYuGtsLTzY6B9QNLXItgxiScf+jkvvTOSpRlCmdtzq\nkoqIoE6b9QCifwa2Fi/cs9GrDUEUi/iz8jaqPoJhfrXU3eqMaNL3MTtiXvctDcHn\nJo7XsLU4Av9aS8QN+67SY4DQCXw9Kloyp0EKkDH5xwKBgQDwIKzd0jIW4d1ugWJq\nYLl7MRkZuTUm3o+2JQKBPEgvmEoJoDt4h/ZNMTdmo+XGWC+/BhPvvnvn4obCF3OP\niaALbFalSvTS5QoKzDnHV/l9ZEpk5dpNd5Vn8NWd3yCM7jWOcYDjAn5I5sEm85Be\nl9d20iAWthnIPbDp2Np2EiT1awKBgQDBHIlLyStAguQx2yCKOiNgHVms5420XS+j\n969GQFGfZARD0F2Rf+yAXTsu5kcM6Gp2fe5eJIOn1apIfqjahg5MPUo7q+sWjtRt\nVKmA4koOggjudYOcSW06f3iOE4xAePBK9+J7hJugAiyXD1HNqkE+D3ENKcft19AQ\nwkxInrjnbwKBgQDpWWeD3IsOj4mGpLdF1x8IZ0sUI1ZSon+XqtmHS1R+5Ag22H5S\ngBXLJ+PFm8pj+DjV8osXNM3mJs17+hwzxbNAxpRg5rmJ5Efg/Fu9q3Fo+DgPWwrM\ns0P+kRyV4UoZijeDaCuu7zJXl97mAlUuh3I8JrBGQcpGPCUa6sBJcxJ1ZwKBgGB7\nyX07/Yg13Z2rRg7KDXKwN2XUK1C6XlsmHUSUTjO83QSkzpsrtxZLfo5oL4ebd9XM\nBZSz2bO5ZWLjJapI4EvnM3es5cBXjHszmZzzctzcy2mY/TDQ3uojVjBmQ+TSh/xs\n7ZOZJchETdMLrGt9bSt8u5dAEMwcz7AP491EsE2xAoGBAM3Vj2E+b/OIntQ4MENd\nLQ4SIgBRmxJpCPN07NagpKocKVet3OL8W6goDuT6gE+VGc10qaYnGG0mVhxp0+kb\ndOJnCh3EHmZJyjTWpKdlxGaiaxjIIxp5PecoQBM9br0HsEvj3B8yVLPEB056/X5p\n8F/8pkLjN6YfEUhsuSFvZ/Pz\n-----END PRIVATE KEY-----\n",
        "client_email": "sk-community@suara-kami-271607.iam.gserviceaccount.com",
        "client_id": "102245550856807249082",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/sk-community%40suara-kami-271607.iam.gserviceaccount.com"
    }
    credentials_info = json.loads(json.dumps(config))
    _credentials = service_account.Credentials.from_service_account_info(
        credentials_info
    )
    project = credentials_info.get("project_id")
    import google.auth
    credentials = google.auth.credentials.with_scopes_if_required(
        _credentials, None
    )
    client = Client(credentials=credentials,project=project)

    file = path.open("rb")
    # print("uploading",path)
    try:
        with open(f'gs://sk-community/{int(info_path.read_text())}/{datetime.now().strftime("%Y%m%d")}/{path.suffix}/{uuid.uuid4()}-{path.name}', 'wb',transport_params=dict(client=client)) as fout:
            fout.write(file.read())
    except Exception as e:
        print(e)
        print("upload failed, please file an issue if needed.")

def compress(path):
    print("compressing",path)
    try:
        shutil.make_archive(path.name, 'zip', path)
    except:
        print("compress failed")
        return ""
    return path.name+".zip"

def feedback(path):
    path = Path(path)
    if path.is_dir():
        path = compress(path)
    upload(path)
    return path

if __name__ == "__main__":
    import fire
    fire.Fire(feedback)
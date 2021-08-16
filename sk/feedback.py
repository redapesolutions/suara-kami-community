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
import base64

info_path = Path.home()/".sk/info.txt"

def upload(path):
    path = Path(path)
    config = "ewogICAgICAgICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgICAgICAgInByb2plY3RfaWQiOiAic3VhcmEta2FtaS0yNzE2MDciLAogICAgICAgICJwcml2YXRlX2tleV9pZCI6ICI5YjU5YzhjYWViZjI0ODI2MzZkNDljMGYwZDI3MjRiYjE5NGRkYjFkIiwKICAgICAgICAicHJpdmF0ZV9rZXkiOiAiLS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tXG5NSUlFdmdJQkFEQU5CZ2txaGtpRzl3MEJBUUVGQUFTQ0JLZ3dnZ1NrQWdFQUFvSUJBUUMxSTJhdXRvSlU3bEMyXG4yS0RzMFlpcE1kNG51dlV1dXI3cjZRWW9qYmYzYmpTUWVDY2paSUwxTUhueEU2L040UHNtN3VkT1YvZVU3WTBSXG5hVzlvNGFZRFdTWnA3TnpWRXJPdmg5RXpscHZhdzlreWsrWERtb3BGd3VaaTNZMlRIZmVPOTU3TWZjSXVpcXFBXG5pRForVDczbFpkVCtlc1RYZVltTVNYVHl4R1cvMU90M25selRZdFJkdmcrdmpMSWpsdGZsaU9ZTWYwbkVBY2NGXG5zNDdsMlc5ZDE4YXBqNnRxTmZ0WGdMNE16K3hhR3VZNEZMRGlzSUFRa3BSOVRMK2NVZlRRT3hqZ29YVWVCRUhRXG5PTnZ4SWZONTFPc09KRTJPdUFNajhmcExsOEs3QVVnVmFpMGJrMEFqK1lyYmFISWVnTlRSTmRpdm9xRWZTclNOXG41N0dtWWZabEFnTUJBQUVDZ2dFQUFZWExqeEtkZW5Qamx6UkpLNTROU2lMUFJ2VTRkMmFYM2crWnNpYjVPRGE5XG5QaTAvbGl6Z2E3MEQ0V1hPUkdxc29jc002b0pnTVJ2TFNrY1VVRjU4YW5kanJpemRwRnJobHpHa2xBeUpqZHEzXG56V0dKNHpQeFFwMFFHaDROUVJ3OEU4eXVkNzc4NFdIZ0JUekdTVXJ1RzU0TG5zN095UHlCcC8ySTRxSHdOSjhYXG4xSUxiN0NsVlA3UEI3UUhlUU9aNXUveVl1R3RzTFR6WTZCOVFOTFhJdGd4aVNjZitqa3Z2VE9TcFJsQ21kdHpxXG5rb3FJb0U2YjlRQ2lmd2EyRmkvY3M5R3JEVUVVaS9pejhqYXFQb0poZnJYVTNlcU1hTkwzTVR0aVh2Y3REY0huXG5KbzdYc0xVNEF2OWFTOFFOKzY3U1k0RFFDWHc5S2xveXAwRUtrREg1eHdLQmdRRHdJS3pkMGpJVzRkMXVnV0pxXG5ZTGw3TVJrWnVUVW0zbysySlFLQlBFZ3ZtRW9Kb0R0NGgvWk5NVGRtbytYR1dDKy9CaFB2dm52bjRvYkNGM09QXG5pYUFMYkZhbFN2VFM1UW9LekRuSFYvbDlaRXBrNWRwTmQ1Vm44TldkM3lDTTdqV09jWURqQW41STVzRW04NUJlXG5sOWQyMGlBV3RobklQYkRwMk5wMkVpVDFhd0tCZ1FEQkhJbEx5U3RBZ3VReDJ5Q0tPaU5nSFZtczU0MjBYUytqXG45NjlHUUZHZlpBUkQwRjJSZit5QVhUc3U1a2NNNkdwMmZlNWVKSU9uMWFwSWZxamFoZzVNUFVvN3Erc1dqdFJ0XG5WS21BNGtvT2dnanVkWU9jU1cwNmYzaU9FNHhBZVBCSzkrSjdoSnVnQWl5WEQxSE5xa0UrRDNFTktjZnQxOUFRXG53a3hJbnJqbmJ3S0JnUURwV1dlRDNJc09qNG1HcExkRjF4OElaMHNVSTFaU29uK1hxdG1IUzFSKzVBZzIySDVTXG5nQlhMSitQRm04cGorRGpWOG9zWE5NM21KczE3K2h3enhiTkF4cFJnNXJtSjVFZmcvRnU5cTNGbytEZ1BXd3JNXG5zMFAra1J5VjRVb1ppamVEYUN1dTd6SlhsOTdtQWxVdWgzSThKckJHUWNwR1BDVWE2c0JKY3hKMVp3S0JnR0I3XG55WDA3L1lnMTNaMnJSZzdLRFhLd04yWFVLMUM2WGxzbUhVU1VUak84M1FTa3pwc3J0eFpMZm81b0w0ZWJkOVhNXG5CWlN6MmJPNVpXTGpKYXBJNEV2bk0zZXM1Y0JYakhzem1aenpjdHpjeTJtWS9URFEzdW9qVmpCbVErVFNoL3hzXG43Wk9aSmNoRVRkTUxyR3Q5YlN0OHU1ZEFFTXdjejdBUDQ5MUVzRTJ4QW9HQkFNM1ZqMkUrYi9PSW50UTRNRU5kXG5MUTRTSWdCUm14SnBDUE4wN05hZ3BLb2NLVmV0M09MOFc2Z29EdVQ2Z0UrVkdjMTBxYVluR0cwbVZoeHAwK2tiXG5kT0puQ2gzRUhtWkp5alRXcEtkbHhHYWlheGpJSXhwNVBlY29RQk05YnIwSHNFdmozQjh5VkxQRUIwNTYvWDVwXG44Ri84cGtMak42WWZFVWhzdVNGdlovUHpcbi0tLS0tRU5EIFBSSVZBVEUgS0VZLS0tLS1cbiIsCiAgICAgICAgImNsaWVudF9lbWFpbCI6ICJzay1jb21tdW5pdHlAc3VhcmEta2FtaS0yNzE2MDcuaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLAogICAgICAgICJjbGllbnRfaWQiOiAiMTAyMjQ1NTUwODU2ODA3MjQ5MDgyIiwKICAgICAgICAiYXV0aF91cmkiOiAiaHR0cHM6Ly9hY2NvdW50cy5nb29nbGUuY29tL28vb2F1dGgyL2F1dGgiLAogICAgICAgICJ0b2tlbl91cmkiOiAiaHR0cHM6Ly9vYXV0aDIuZ29vZ2xlYXBpcy5jb20vdG9rZW4iLAogICAgICAgICJhdXRoX3Byb3ZpZGVyX3g1MDlfY2VydF91cmwiOiAiaHR0cHM6Ly93d3cuZ29vZ2xlYXBpcy5jb20vb2F1dGgyL3YxL2NlcnRzIiwKICAgICAgICAiY2xpZW50X3g1MDlfY2VydF91cmwiOiAiaHR0cHM6Ly93d3cuZ29vZ2xlYXBpcy5jb20vcm9ib3QvdjEvbWV0YWRhdGEveDUwOS9zay1jb21tdW5pdHklNDBzdWFyYS1rYW1pLTI3MTYwNy5pYW0uZ3NlcnZpY2VhY2NvdW50LmNvbSIKICAgIH0="
    credentials_info = json.loads(base64.b64decode(config)) # create only perm
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
        with open(f'gs://sk-community/{int(info_path.read_text())}/{datetime.now().strftime("%Y%m%d")}/{path.suffix}/{uuid.uuid4()}-sk-{path.name}', 'wb',transport_params=dict(client=client)) as fout:
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
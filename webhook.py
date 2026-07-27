import json
from pathlib import Path

import requests

import base64
import mimetypes
import requests
import json
from pathlib import Path

import os
from dotenv import load_dotenv

class WebhookBuilder:
    load_dotenv()

    AVATAR_URL_PATH = Path("discord") / "avatar_url.txt"
    WEBHOOK_URL = os.getenv("WEBHOOK_URL")
    USERNAME = "Stichy"

    def __init__(self):
        self.index = 0
        with self.AVATAR_URL_PATH.open() as f:
            self.avatar_urls = [line.strip() for line in f if line.strip()]
            
    def update_profile(self, name=None, avatar=None):
        payload = {}
        if name is not None:
            payload["name"] = name
        if avatar is not None:
            with open(avatar, "rb") as img_file:
                img = base64.b64encode(img_file.read()).decode("utf-8")
            mt = mimetypes.guess_type(avatar)
            mt = mt if mt else "image/png"
            payload["avatar"] = f"data:{mt};base64,{img}"
        requests.patch(self.WEBHOOK_URL, json=payload)

    def send_embed(self, title=None, description=None, image=None):
        embed = {}
        if title is not None:
            embed["title"] = title
        if description is not None:
            embed["description"] = description
        if image is None:
            print(embed)
            r = requests.post(self.WEBHOOK_URL, json={"embeds": [embed]})
        else:
            extension = Path(image).suffix
            embed["image"] = {"url": f"attachment://attachment{extension}"}
            mt = mimetypes.guess_type(image)[0]
            mt = mt if mt else "image/png"
            with open(image, "rb") as img_file:
                files = {
                    "payload_json": (None, json.dumps({"embeds": [embed]})),
                    "file": (f"attachment{extension}", img_file, mt),
                }
                r = requests.post(self.WEBHOOK_URL, files=files, json=embed)

    def post(self, title, description, plot):
        embed_payload = {
            "username": self.USERNAME,
            "avatar_url": self.avatar_urls[self.index % len(self.avatar_urls)],
            "embeds": [
                {
                    "title": title,
                    "description": description,
                    "image": {
                        "url": (
                            f"attachment://{Path(plot).name}"
                            if not str(plot).startswith(("http://", "https://"))
                            else plot
                        )
                    }
                }
            ]
        }

        # Local file -> upload it
        if not str(plot).startswith(("http://", "https://")):
            with open(plot, "rb") as f:
                requests.post(
                    self.WEBHOOK_URL,
                    data={
                        "payload_json": json.dumps(embed_payload)
                    },
                    files={
                        "file": (Path(plot).name, f)
                    }
                )
        # Remote URL -> just send JSON
        else:
            requests.post(
                self.WEBHOOK_URL,
                json=embed_payload
            )
            
        self.index += 1
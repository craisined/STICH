import json
from pathlib import Path

import requests

class WebhookBuilder:

    AVATAR_URL_PATH = Path("discord") / "avatar_url.txt"
    WEBHOOK_URL = "https://discord.com/api/webhooks/1531124590363803848/nqf6ZRkZX0S8Ba_ZhuE4vXeDq-3nvOQfQDrQ9DoGMjqVJeiuyR4c8QUDVTe1qDk5Fbx_"
    USERNAME = "Stichy"

    def __init__(self):
        self.index = 0
        with self.AVATAR_URL_PATH.open() as f:
            self.avatar_urls = [line.strip() for line in f if line.strip()]

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
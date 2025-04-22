"""Implementation of automated moderator"""

from typing import List
from atproto import Client
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
from dotenv import load_dotenv
import imagehash
import os


T_AND_S_LABEL = "t-and-s"
DOG_LABEL = "dog"
THRESH = 17

class AutomatedLabeler:
    """Automated labeler implementation"""

    def __init__(self, client: Client, input_dir):
        self.client = client
        self.input_dir = input_dir
        self.urls = None

        try:
            self.word_df = pd.read_csv(os.path.join(self.input_dir, "t-and-s-words.csv"))
            self.domain_df = pd.read_csv(os.path.join(self.input_dir, "t-and-s-domains.csv"))
        except Exception as e:
            print(f"[ERROR] Failed to load input CSVs from {self.input_dir}: {e}")

        self.dog_hashes = []
        dog_image_dir = os.path.join(self.input_dir, "dog-list-images")
        if os.path.exists(dog_image_dir):
            self.load_dog_image_hashes(dog_image_dir)
        if dog_image_dir:
            self.load_dog_image_hashes(dog_image_dir)

        try:
            self.read_urls()
        except Exception as e:
            pass

    def url_get_post(self, url):
        parts = url.split("/")
        rkey = parts[-1]
        handle = parts[-3]
        post = self.client.get_post(rkey, handle)
        return post

    def moderate_post(self, url: str) -> List[str]:
        labels = []

        try:
            post = self.url_get_post(url)
            if self.label_words(post):
                labels.append(T_AND_S_LABEL)
        except Exception as e:
            print(f"[WARNING] Error in text moderation for {url}: {e}")

        try:
            img = self.get_image(url)
            if img and self.image_matches_dog(img):
                labels.append(DOG_LABEL)
        except Exception as e:
            print(f"[INFO] No valid image found or failed to match dog image for {url}: {e}")

        return list(set(labels))

    def read_urls(self):
        if self.input_dir.endswith(".csv"):
            df = pd.read_csv(self.input_dir)
            self.urls = df["URL"]

    def label_words(self, post):
        word_ls = self.word_df["Word"].tolist()
        word_ls_lower = [word.lower() for word in word_ls]
        domains_ls = self.domain_df["Domain"].tolist()
        domains_ls_lower = [domain.lower() for domain in domains_ls]

        text = post.value.text
        text_lower = text.lower()
        if any(elem in text_lower for elem in word_ls_lower) or any(elem in text_lower for elem in domains_ls_lower):
            return ["t-and-s"]
        else:
            return []
    
    def get_image(self, url):
        post = self.url_get_post(url)
        uri = post.uri
        thread = self.client.get_post_thread(uri)
        fullsize = thread.thread.post.embed.images[0].fullsize
        response = requests.get(fullsize)
        if response.status_code == 200:
            # Open the image using PIL
            img = Image.open(BytesIO(response.content))
        return img

    def load_dog_image_hashes(self, dog_image_dir):
        for fname in os.listdir(dog_image_dir):
            if fname.lower().endswith((".jpg")):
                path = os.path.join(dog_image_dir, fname)
                try:
                    img = Image.open(path)
                    img_hash = imagehash.phash(img)
                    self.dog_hashes.append(img_hash)
                except Exception as e:
                    print(f"Error hashing {fname}: {e}")

    def image_matches_dog(self, img):
        img_hash = imagehash.phash(img)
        for ref_hash in self.dog_hashes:
            if img_hash - ref_hash <= THRESH:
                return True
        return False



load_dotenv(override=True)
USERNAME = "trustandsafety14.bsky.social"#os.getenv("USERNAME")
PW = "W2DU:TG3fwn9dgc"#os.getenv("PW")
client = Client()
client.login(USERNAME, PW)


### Milestone 1 Label t&s words and domains
def t_and_s_word_domain_label():
    word_labeler = AutomatedLabeler(client, "../test-data/input-posts-t-and-s.csv")
    word_labels = []
    for url in word_labeler.urls:
        word_labels.append(word_labeler.moderate_post(url))
    print(word_labels)

### Milestone 4: Dog Labeling using Perceptual Hash
def dog_labeler():
    dog_labeler = AutomatedLabeler(
        client,
        "../test-data/input-posts-dogs.csv",
        dog_image_dir="../labeler-inputs/dog-list-images"
    )
    results = []
    for url in dog_labeler.urls:
        results.append(dog_labeler.moderate_post(url))
    print(results)



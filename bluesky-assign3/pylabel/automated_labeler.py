"""Implementation of automated moderator"""

from typing import List
from atproto import Client
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
from dotenv import load_dotenv


T_AND_S_LABEL = "t-and-s"
DOG_LABEL = "dog"
THRESH = 0.3

class AutomatedLabeler:
    """Automated labeler implementation"""

    def __init__(self, client: Client, input_dir):
        self.client = client
        self.input_dir = input_dir
        self.urls = None
        self.read_urls()

    def url_get_post(self, url):
        parts = url.split("/")
        rkey = parts[-1]
        handle = parts[-3]
        post = self.client.get_post(rkey, handle)
        return post
   
    def moderate_post(self, url: str) -> List[str]:
        """
        Apply moderation to the post specified by the given url
        """
        post = self.url_get_post(url)
        label = self.label_words(post)
        result = [url, label]
        return label
    
    def read_urls(self):
        df = pd.read_csv(self.input_dir)
        self.urls = df["URL"]

    def label_words(self, post):
        try:
            df = pd.read_csv("labeler-inputs/t-and-s-words.csv")
            df1 = pd.read_csv("labeler-inputs/t-and-s-domains.csv")
        except Exception as e:
            print(e)
        word_ls = df["Word"].tolist()
        word_ls_lower = [word.lower() for word in word_ls]
        domains_ls = df1["Domain"].tolist()
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


### Test get image from url
# image_test = AutomatedLabeler(client, "../test-data/input-posts-dogs.csv")
# image = image_test.get_image(image_test.urls[0])
# image.show()
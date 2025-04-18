"""Implementation of automated moderator"""

from typing import List
from atproto import Client
import pandas as pd
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
        self.post = []
        self.get_post()
   
    def moderate_post(self, url: str) -> List[str]:
        """
        Apply moderation to the post specified by the given url
        """
        result = self.label_words()
        return result
    
    def read_urls(self):
        df = pd.read_csv(self.input_dir)
        self.urls = df["URL"]

    def get_post(self):
        for url in self.urls:
            parts = url.split("/")
            rkey = parts[-1]
            handle = parts[-3]
            self.post.append(self.client.get_post(rkey, handle))

    def label_words(self):
        try:
            df = pd.read_csv("../labeler-inputs/t-and-s-words.csv")
            df1 = pd.read_csv("../labeler-inputs/t-and-s-domains.csv")
        except Exception as e:
            print(e)
        word_ls = df["Word"].tolist()
        word_ls_lower = [word.lower() for word in word_ls]
        domains_ls = df1["Domain"].tolist()
        domains_ls_lower = [domain.lower() for domain in domains_ls]
        result = []

        for i, p in enumerate(self.post):
            text = p.value.text
            text_lower = text.lower()
            if any(elem in text_lower for elem in word_ls_lower) or any(elem in text_lower for elem in domains_ls_lower):
                result.append(self.urls[i])

        return result


load_dotenv(override=True)
USERNAME = "trustandsafety14.bsky.social"#os.getenv("USERNAME")
PW = "W2DU:TG3fwn9dgc"#os.getenv("PW")
client = Client()
client.login(USERNAME, PW)
word_labeler = AutomatedLabeler(client, "../test-data/input-posts-t-and-s.csv")
print(word_labeler.moderate_post(""))

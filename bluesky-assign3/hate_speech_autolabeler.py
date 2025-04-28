import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pytesseract
from PIL import Image
from io import BytesIO
import requests
import cv2
from atproto import Client
import os
from urllib.parse import urljoin
import tempfile
import re
from tqdm import tqdm
from dotenv import load_dotenv


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    return " ".join(words)


class HateSpeechClassifier:
    """A wrapper class for using the trained hate speech classification model"""
    
    def __init__(self, model_path="hate_speech_classifier"):
        """
        Initialize the classifier with a trained model
        
        Args:
            model_path: Path to the saved model directory
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define class labels
        self.labels = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
    
    def predict(self, text):
        """
        Classify a single text
        
        Args:
            text: The text to classify
            
        Returns:
            A dictionary with the predicted class and confidence scores
        """
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Get prediction and confidence
            pred_class = torch.argmax(logits, dim=1).item()
        
        labels = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
        
        return labels[pred_class]


class HateSpeechLabeler:

    def __init__(self, client: Client):
        self.client = client
        self.classifier = HateSpeechClassifier("hate_speech_classifier")

    def url_get_post(self, url):
        parts = url.split("/")
        rkey = parts[-1]
        handle = parts[-3]
        post = self.client.get_post(rkey, handle)
        return post
    
    def moderate_post(self, url):
        post = self.url_get_post(url)

        try:
            text = post.value.text
            if text and self.label_text(text) == "Hate Speech":
                return ["Hate Speech"]
        except Exception as e:
            print(f"[WARNING] Error in text moderation for {url}: {e}")
        
        try:
            img = self.get_image(post)
            if img and self.label_image(img) == "Hate Speech":
                return ["Hate Speech"]
        except Exception as e:
            print(f"[WARNING] Error in image moderation for {url}: {e}")
        
        try:
            video_url = self.get_video_url(post)
            temp_video = self.process_video(video_url)
            if temp_video and self.label_video(temp_video) == "Hate Speech":
                return ["Hate Speech"]
        except Exception as e:
            print(f"[WARNING] Error in video moderation for {url}: {e}")

        return []
    
    def get_video_url(self, post):
        uri = post.uri
        thread = self.client.get_post_thread(uri)
        return thread.thread.post.embed.playlist
    
    def get_image(self, post):
        uri = post.uri
        thread = self.client.get_post_thread(uri)
        fullsize = thread.thread.post.embed.images[0].fullsize
        response = requests.get(fullsize)
        if response.status_code == 200:
            # Open the image using PIL
            img = Image.open(BytesIO(response.content))
        return img
    
    def process_video(self, master_m3u8_url):
        temp_dir = tempfile.mkdtemp()

        # Get the master playlist
        response = requests.get(master_m3u8_url)
        if response.status_code != 200:
            print(f"Failed to download master playlist: {response.status_code}")
            return None
        
        # Parse the master playlist
        master_content = response.text
        print("Master playlist content:")
        print(master_content)
        
        # Get the base URL for the master playlist
        base_url = master_m3u8_url.rsplit('/', 1)[0] + '/'
        
        # Find all stream options using regex
        stream_pattern = r'#EXT-X-STREAM-INF:.*?RESOLUTION=(\d+x\d+).*?\n(.*?)$'
        streams = re.findall(stream_pattern, master_content, re.MULTILINE)
        
        if not streams:
            print("No streams found in the master playlist")
            return None
        
        # Select the highest resolution stream
        highest_res = 0
        highest_stream = None
        
        for resolution, stream_path in streams:
            width, height = map(int, resolution.split('x'))
            res = width * height
            if res > highest_res:
                highest_res = res
                highest_stream = stream_path
        
        if not highest_stream:
            print("Could not determine highest quality stream")
            return None
        
        # Construct the full URL for the selected stream
        stream_url = urljoin(base_url, highest_stream)
        print(f"Selected stream: {stream_url}")
        
        # Get the stream playlist
        response = requests.get(stream_url)
        if response.status_code != 200:
            print(f"Failed to download stream playlist: {response.status_code}")
            return None
        
        # Parse the stream playlist
        stream_content = response.text
        print("Stream playlist content preview:")
        print(stream_content[:500] + "..." if len(stream_content) > 500 else stream_content)
        
        # Extract segment URLs
        segments = []
        stream_base_url = stream_url.rsplit('/', 1)[0] + '/'
        
        for line in stream_content.splitlines():
            if not line.startswith('#') and line.strip():
                # This is a segment file
                segment_url = urljoin(stream_base_url, line)
                segments.append(segment_url)
        
        if not segments:
            print("No segments found in the stream playlist")
            return None
        
        print(f"Found {len(segments)} video segments")
        
        # Download and process segments
        temp_video = os.path.join(temp_dir, "combined_video.ts")
        
        with open(temp_video, 'wb') as outfile:
            for i, segment_url in enumerate(tqdm(segments)):
                try:
                    response = requests.get(segment_url)
                    if response.status_code == 200:
                        outfile.write(response.content)
                    else:
                        print(f"Failed to download segment {i}: {response.status_code}")
                except Exception as e:
                    print(f"Error downloading segment {i}: {str(e)}")
        
        return temp_video

    def label_text(self, text):
        cleaned_text = preprocess(text)
        label = self.classifier.predict(cleaned_text)
        return label
    
    def label_image(self, img):
        text = pytesseract.image_to_string(img)
        label = self.label_text(text)
        return label

    def label_video(self, temp_video):
        cap = cv2.VideoCapture(temp_video)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video contains {frame_count} frames at {fps} FPS")
        
        # Process frames - take one frame per second to avoid excessive processing
        frames_to_sample = max(1, int(fps))  # Take at least one frame per second
        results = ""
        
        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process every Nth frame (1 per second)
            if current_frame % frames_to_sample == 0:
                # Convert to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply some preprocessing to improve OCR
                # Increase contrast
                gray_frame = cv2.convertScaleAbs(gray_frame, alpha=1.5, beta=0)
                
                # Apply threshold to make text more visible
                _, threshold = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)
                
                # Extract text
                text = pytesseract.image_to_string(threshold)
                
                # Only add if text was found
                if text.strip():
                    results += text.strip() + " "
            
            current_frame += 1
            if current_frame % 100 == 0:
                print(f"Processed {current_frame} frames...")
        
        cap.release()
        print(f"Completed video processing. Found text in {len(results)} frames.")
        label = self.label_text(results)

        return label


# Example usage in another script
if __name__ == "__main__":
    load_dotenv(override=True)
    USERNAME = "trustandsafety14.bsky.social"#os.getenv("USERNAME")
    PW = "W2DU:TG3fwn9dgc"#os.getenv("PW")
    client = Client()
    client.login(USERNAME, PW)

    autolabeler =  HateSpeechLabeler(client)
    url = "https://bsky.app/profile/idefixasterix.bsky.social/post/3lbgxvicics2h"
    label = autolabeler.moderate_post(url)

    # ## Extract text from image
    # image = Image.open('ex3.jpg')
    # text = pytesseract.image_to_string(image)
    
    # # Classify a single text
    # result = classifier.predict(text)
    # #print(f"Prediction: {result['predicted_label']}")
    # #print(f"Confidence scores: {result['confidence_scores']}")
    # print(result)
    print(label)
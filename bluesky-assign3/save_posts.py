import pickle
import os
import datetime as datetime

class PostStorage:
    """Class to manage storing and retrieving multiple posts"""
    
    def __init__(self, filename="saved_posts.pkl"):
        """
        Initialize the storage with a filename
        
        Args:
            filename: The pickle file to use for storage
        """
        self.filename = filename
        self.posts = {}
        self._load()
    
    def _load(self):
        """Load posts from the pickle file if it exists"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'rb') as f:
                    self.posts = pickle.load(f)
                print(f"Loaded {len(self.posts)} posts from {self.filename}")
            except Exception as e:
                print(f"Error loading posts: {e}")
                self.posts = {}
    
    def save(self):
        """Save all posts to the pickle file"""
        with open(self.filename, 'wb') as f:
            pickle.dump(self.posts, f)
        print(f"Saved {len(self.posts)} posts to {self.filename}")
    
    def add_post(self, post_id, thread, url):
        """
        Add a post to the storage
        
        Args:
            post_id: Unique identifier for the post 
                    (could be post.uri or any other unique ID)
            thread: The Thread object
            url: The fullsize URL or any other URL to store
        """
        self.posts[post_id] = {
            "thread": thread,
            "url": url,
            "added_at": datetime.now()
        }
        self.save()
        return True
    
    def get_post(self, post_id):
        """
        Retrieve a post by its ID
        
        Args:
            post_id: The ID of the post to retrieve
        
        Returns:
            dict: Post data including thread and URL, or None if not found
        """
        return self.posts.get(post_id)
    
    def remove_post(self, post_id):
        """
        Remove a post from storage
        
        Args:
            post_id: The ID of the post to remove
            
        Returns:
            bool: True if post was removed, False if not found
        """
        if post_id in self.posts:
            del self.posts[post_id]
            self.save()
            return True
        return False
    
    def list_posts(self):
        """
        List all stored posts
        
        Returns:
            list: List of post IDs and when they were added
        """
        return [(post_id, data["added_at"]) for post_id, data in self.posts.items()]
    
    def get_all_posts(self):
        """
        Get all stored posts
        
        Returns:
            dict: All posts
        """
        return self.posts

storage = PostStorage()

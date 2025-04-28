import requests
import csv
import time

USERNAME = "trustandsafety14.bsky.social"
PASSWORD = "W2DU:TG3fwn9dgc"

def login(username, password):
    url = "https://bsky.social/xrpc/com.atproto.server.createSession"
    payload = {"identifier": username, "password": password}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["accessJwt"]


def search_posts(token, total_limit=10000, keyword=" "):
    url = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
    headers = {"Authorization": f"Bearer {token}"}
    all_posts = []
    cursor = None

    while len(all_posts) < total_limit:
        params = {
            "limit": 100,
            "q": keyword,
        }
        if cursor:
            params["cursor"] = cursor

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        batch = data.get("posts", [])
        all_posts.extend(batch)

        print(f"Fetched {len(all_posts)} posts so far...")

        cursor = data.get("cursor")
        if not cursor:
            print("No more pages available.")
            break

        time.sleep(0.5)

    return {"posts": all_posts[:total_limit]}


def filter_intolerance_posts(posts_data):
    intolerance_posts = []
    for post in posts_data.get("posts", []):
        labels = post.get("labels", [])
        if labels:
            for label in labels:
                if "intolerant" in label.get("val", ""):
                    intolerance_posts.append({
                        "uri": post.get("uri", ""),
                        "text": post.get("record", {}).get("text", ""),
                        "label": label.get("val", ""),
                        "createdAt": post.get("indexedAt", "")
                    })
    return intolerance_posts

def save_to_csv(posts, filename="intolerance_searchposts.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["URI", "Text", "Label", "Created At"])
        for post in posts:
            writer.writerow([post["uri"], post["text"], post["label"], post["createdAt"]])

def main():
    try:
        print("Logging in...")
        token = login(USERNAME, PASSWORD)
        print("Login successful")

        print("Searching posts...")
        posts_data = search_posts(token, total_limit=10000, keyword="nigger")

        print("Filtering posts with Intolerance labels...")
        intolerance_posts = filter_intolerance_posts(posts_data)

        print(f"Found {len(intolerance_posts)} intolerance-labeled posts.")
        for post in intolerance_posts:
            print(f"{post['uri']} | {post['label']} | {post['createdAt']}")

        save_to_csv(intolerance_posts)
        print("Saved to intolerance_searchposts.csv")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
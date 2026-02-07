import requests
import json
import os
from dotenv import load_dotenv
import random
import time
import math

load_dotenv()
token = os.getenv("GITHUB_TOKEN")

headers ={
      'Authorization': f'token {token}',
    }

# get my user ID
USERNAME = 'sarayuv'
resp = requests.get(f"https://api.github.com/users/{USERNAME}")
resp.raise_for_status()
user = resp.json()
print(user["login"], user["id"])

# collect data by users API
id_ = 0
response = requests.get('https://api.github.com/users?since='+str(id_),headers=headers)
data = response.json()

# collect data by search API
response = requests.get('https://api.github.com/search/users?q=created:>=2025-01-22&sort=joined&order=desc',headers=headers)
data = response.json()

json_formatted_str = json.dumps(data, indent=2)
print(json_formatted_str)

# It will return 30 results for each request. You could consider using "for" loop to crawl more data.
# The sample code is a simple way to collect GitHub users' ID. You can consider other ways to collect data.
""" 
{'login': 'mojombo',
     'id': 1,
     'node_id': 'MDQ6VXNlcjE=',
     'avatar_url': 'https://avatars0.githubusercontent.com/u/1?v=4',
     'gravatar_id': '',
     'url': 'https://api.github.com/users/mojombo',
     'html_url': 'https://github.com/mojombo',
     'followers_url': 'https://api.github.com/users/mojombo/followers',
     'following_url': 'https://api.github.com/users/mojombo/following{/other_user}',
     'gists_url': 'https://api.github.com/users/mojombo/gists{/gist_id}',
     'starred_url': 'https://api.github.com/users/mojombo/starred{/owner}{/repo}',
     'subscriptions_url': 'https://api.github.com/users/mojombo/subscriptions',
     'organizations_url': 'https://api.github.com/users/mojombo/orgs',
     'repos_url': 'https://api.github.com/users/mojombo/repos',
     'events_url': 'https://api.github.com/users/mojombo/events{/privacy}',
     'received_events_url': 'https://api.github.com/users/mojombo/received_events',
     'type': 'User',
     'site_admin': False} 
"""
     
#sample NUM ids since UID    
def sample(uid,num):
    """
    randomly sample GitHub user IDs and check whether the sampled IDs are valid or not.
    
    :param uid: Description
    :param num: Description

    :return: list of 0 (invalid) or 1 (valid) values
    """

    sample_data = []

    for _ in range(num):
        sampled_id = random.randint(1, uid)

        response = requests.get(f'https://api.github.com/user/{sampled_id}', headers=headers)

        if response.status_code == 200:
            sample_data.append(1)
        else:
            sample_data.append(0)

        time.sleep(0.25)

    return sample_data
    
#use downloaded data to build estimator  
def estimate(sample_data, max_uid):
    """
    estimate the total number of valid users based on the sampled data
    
    :param sample_data: Description
    :param max_uid: Description
    :return: estimation of the total number of valid users
    """

    valid = sum(sample_data)
    total = len(sample_data)
    
    p_hat = valid / total
    estimation = p_hat * max_uid

    print("Sample size:", len(sample_data))
    print("Valid samples:", sum(sample_data))
    print("Estimated total valid users:", int(estimation))

    return estimation

if __name__ == "__main__":
    MAX_UID = 150_000_000
    SAMPLE_SIZE = 500

    print("\nRunning Iteration 1 Sampling")
    sample_data = sample(MAX_UID, SAMPLE_SIZE)

    print("\nEstimation")
    estimate(sample_data, MAX_UID)

    p_hat = sum(sample_data) / len(sample_data)
    n = len(sample_data)

    stderr = math.sqrt(p_hat * (1 - p_hat) / n)
    ci_low = (p_hat - 1.96 * stderr) * MAX_UID
    ci_high = (p_hat + 1.96 * stderr) * MAX_UID

    print(f"\n95% Confidence Interval: [{int(ci_low)}, {int(ci_high)}]")
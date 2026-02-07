import requests
import json

headers ={
      'Authorization': 'token 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',
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
     
#sample NUM ids since UID    
def sample(uid,num):
    ...
    return sample_data
    
#use downloaded data to build estimator  
def estimate(sample_data):
    ...
    return estimation
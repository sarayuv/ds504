import requests
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
     
def sample(uid, num):
    """
    randomly sample GitHub user IDs and check whether the sampled IDs are valid or not.
    
    :param uid: maximum user ID to sample from
    :param num: number of samples to draw

    :return: list of 0 (invalid) or 1 (valid) values
    """

    sample_data = []

    for _ in range(num):
        sampled_id = random.randint(1, uid)

        response = requests.get(f'https://api.github.com/user/{sampled_id}', headers=headers)

        # check if the sampled ID is valid or not
        if response.status_code == 200:
            sample_data.append(1)
        else:
            sample_data.append(0)

        time.sleep(0.25)

    return sample_data
    

def estimate(sample_data, max_uid):
    """
    estimate the total number of valid users based on the sampled data
    
    :param sample_data: list of 0 (invalid) or 1 (valid) values
    :param max_uid: maximum user ID considered in the sampling
    :return: estimation of the total number of valid users
    """

    p_hat = sum(sample_data) / len(sample_data)
    estimation = p_hat * max_uid

    print("Sample size:", len(sample_data))
    print("Valid samples:", sum(sample_data))
    print("Estimated total valid users:", int(estimation))

    return estimation


if __name__ == "__main__":
    # parameters
    MAX_UID = 150_000_000
    SAMPLE_SIZE = 500

    print("\nRunning Iteration 1 Sampling")
    sample_data = sample(MAX_UID, SAMPLE_SIZE)

    print("\nEstimation")
    estimate(sample_data, MAX_UID)

    p_hat = sum(sample_data) / len(sample_data)
    n = len(sample_data)
    
    # calculate standard error and confidence interval
    stderr = math.sqrt(p_hat * (1 - p_hat) / n)
    ci_low = (p_hat - 1.96 * stderr) * MAX_UID
    ci_high = (p_hat + 1.96 * stderr) * MAX_UID
    print(f"\n95% Confidence Interval: [{int(ci_low)}, {int(ci_high)}]")
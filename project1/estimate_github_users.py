import requests
import os
from dotenv import load_dotenv
import random
import time
import matplotlib.pyplot as plt
import statistics
import csv

# authentication
load_dotenv()
token = os.getenv("GITHUB_TOKEN")
headers = {
      'Authorization': f'token {token}',
    }

# parameters
VALIDATION_MAX_ID = 10000
GLOBAL_MAX_ID = 150000000
SAMPLING_BUDGETS = [500, 1000, 2000, 5000]
RUNS_PER_BUDGET = 10
REQUEST_DELAY = 0.25


def build_validation_set(max_id):
    """
    Exhaustively check the validity of user IDs in ID range [1, max_id] and build a validation set.
    
    :param max_id: maximum user ID to check
    """

    valid_users = set()

    for uid in range(1, max_id + 1):
        response = requests.get(f'https://api.github.com/user/{uid}', headers=headers)

        if response.status_code == 200:
            valid_users.add(uid)

        time.sleep(REQUEST_DELAY)

    print(f"\n{len(valid_users)} valid users found.")

    return valid_users


def sample(uid, num, valid_set=None):
    """
    randomly sample GitHub user IDs and check whether the sampled IDs are valid or not.
    
    :param uid: maximum user ID to sample from
    :param num: number of samples to draw
    :param valid_set: set of valid user IDs for validation

    :return: list of 0 (invalid) or 1 (valid) values
    """

    sample_data = []

    for _ in range(num):
        sampled_id = random.randint(1, uid)

        if valid_set is not None:
            sample_data.append(1 if sampled_id in valid_set else 0)
        else:
            response = requests.get(f'https://api.github.com/user/{sampled_id}', headers=headers)
            sample_data.append(1 if response.status_code == 200 else 0)
            time.sleep(REQUEST_DELAY)

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

    return estimation


if __name__ == "__main__":
    # BUILD VALIDATION SET
    # store valid IDs in the validation range
    validation_valid_users = build_validation_set(VALIDATION_MAX_ID)
    # number of valid IDs in validation set
    ground_truth = len(validation_valid_users)

    # RUN SAMPLING
    # initialize dict for estimates
    estimates_by_budget = {}

    print("\nRunning Sampling and Estimation")
    for budget in SAMPLING_BUDGETS:
        estimates = []

        for run in range(RUNS_PER_BUDGET):
            # perform sampling and estimation using the validation set
            sample_data = sample(VALIDATION_MAX_ID, budget, valid_set=validation_valid_users)
            # estimate total valid users based on sampled data
            est = estimate(sample_data, VALIDATION_MAX_ID)
            estimates.append(est)

            print(f"Budget: {budget}, Run: {run + 1}, Estimate: {int(est)}")
    
        # store estimates for the current budget
        estimates_by_budget[budget] = estimates

    # CREATE TABLE
    summary_rows = []

    print("\nSummary Table:")
    print(f"{'Budget':<10} {'Mean Estimate':<20} {'Std Dev':<15} {'Ground Truth':<15}")

    for budget, estimates in estimates_by_budget.items():
        mean_estimate = statistics.mean(estimates)
        std_dev = statistics.stdev(estimates)
        summary_rows.append([budget, mean_estimate, std_dev, ground_truth])

        print(f"{budget:<10} {mean_estimate:<20.2f} {std_dev:<15.2f} {ground_truth:<15}")

    # save table to CSV
    with open('estimation_summary.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Budget', 'Mean Estimate', 'Std Dev', 'Ground Truth'])
        writer.writerows(summary_rows)

    # PLOT
    for budget, estimates in estimates_by_budget.items():
        plt.scatter([budget] * len(estimates), estimates, label=f'Budget {budget}')

    plt.axhline(y=ground_truth, color='darkblue', linestyle='--', label='Ground Truth')
    plt.xlabel('Sampling Budget')
    plt.ylabel('Estimated Valid Users')
    plt.title('Estimation of Valid GitHub Users vs Sampling Budget')
    plt.legend()
    plt.show()

    # ESTIMATE FULL ID SPACE
    print("\nEstimating Full ID Space")
    FULL_SAMPLE_SIZE = 10000
    full_sample_data = sample(GLOBAL_MAX_ID, FULL_SAMPLE_SIZE)
    full_estimate = estimate(full_sample_data, GLOBAL_MAX_ID)
    print(f"Estimated total valid users in full ID space: {int(full_estimate)}")
# Individual Project 4
# Meta-Learning and Few Shot Learning
#### Due Date
* Tuesday Apr 28, 2026 (23:59)

#### Total Points
* 100 (One Hundred)

## Goal
In Project 2, you were given a bunch of drivers and their trajectories to build a model to classify which driver a given 100-step length sub-trajectory belongs to. In this project, we will give you a harder task. In Project 2, the training data contains 5 drivers and 6-month trajectories for each driver. In this task, however, the training data contain 400 drivers and only 5-day trajectories for each driver. You should use few-shot learning to build the classification model for this task. The model for each driver can be a binary classification, which takes two trajectories as input and predicts whether these two trajectories belong to the same driver. 

<img src="https://github.com/UrbanIntelligence/CS586-DS504-Spring2024/blob/master/project4/pic/Project4_4.png" width="100%">

## How to run :
extract features from the pickle file and generate the sub-trajectories of the 400 drivers:
* `$ python extract_feature.py`

generate pairs of sub-trajectories where pairs from the same driver are labeled with a 1 and pairs from different drivers are labeled with a 0. Upon executing this script, the expected shape of the training dataset will be (number of trajectory pairs, 2, 100, feature size).

* `$ generate_paired_traj.py`
  
training model:
* `$ python main.py train`

testing model:
* `$ python main.py test --test_dir = `

## Current Leaderboard
| rank | Name | Accuracy |
|---|---|---|
|**1**   | | |
|**2**   | | |
|**3**   | | |


## Evaluation
To evaluate your submission, a separate test dataset will be used. The test data will be another 100 different drivers, and we will use the same data preprocessing as you did for the training dataset. We will randomly generate 10,000 100-step length sub-trajectory pairs and use them to evaluate your submitted model. This means the testing dataset's shape is (10000, 2, 100, feature_size).

## Deliverables & Grading

### Gradescope Assignment Submission Guide

Follow these instructions to correctly prepare and submit your assignment on Gradescope.

#### Code Files
Include all necessary Python scripts for your project:
- `main.py`: Entry point for the project
- `train.py`: Training logic
- `test.py`: Testing logic
- `extract_feature.py`: Data preprocessing logic
- `generate_paired_traj.py`: (number of trajectory pairs, 2, 100, feature size)
- `model.py`: Model definition

#### Trained Model
Include the trained model file (e.g., `best_model.pt`).  
- **Note**: The filename must match the reference in your `test.py` script.

#### Requirements
Include a requirements.txt file that lists any additional Python libraries and dependencies your code requires beyond those already provided.

#### Output Format

- Your `test.py` script must print the **accuracy** in the following format:
    ```plaintext
    Accuracy=0.6123
    ```
  Do not multiply the accuracy by 100; percentages are not allowed.
#### Zip Your Submission

Prepare a ZIP file containing the following files:

- `main.py`: Entry point for the project
- `train.py`: Training logic
- `test.py`: Testing logic
- `extract_feature.py`: Data preprocessing logic
- `generate_paired_traj.py`: (number of trajectory pairs, 2, 100, feature size)
- `model.py`: Model definition
- **trained model file** (e.g., `best_model.pt`)
- (Optional) **`requirements.txt`**: If extra dependencies are needed.
- **PDF Report**

## **Important Notes**:
  1. The accuracy value must be a floating-point number formatted to **4 decimal places**, don't change the line **"print(f"Accuracy={test_accu:.4f}")"** in test.py for the correct output for the Autograder.
  2. The autograder will evaluate your submission on a CPU-only machine. Please handle it in your code.
  3. Don't change line **"test_file_pattern = os.path.join(test_dir, "X_Y_test100_pairs.pkl")"** in test.py for test.
     
#### Grading

* PDF Report (50%) [template](https://www.acm.org/binaries/content/assets/publications/taps/acm_submission_template.docx)
    * proposal
    * Methodology
    * empirical results and evaluation
    * conclusion
    
* Python Code (50%)
    * Code is required to avoid plagiarism.
    * The submission should contain all the Python files including "extract_feature.py, generate_paired_traj.py, model.py, train.py, test.py, and main.py" to help evaluate your model.
    * Similar to Project2, you can revise "extract_feature.py" to get more features of the dataset or for feature engineering. But for a fair evaluation of the performance of the code, the input sub-trajectory length has to be 100.
    * Evaluation criteria.
      | Percentage | Accuracy |
      |---|---|
      | 100 | 0.65 |
      | 90 | 0.60 |
      | 80 | 0.55 |
      | 70 | 0.50|

* Grading:
  * Total (100):
    * Code (50) + Report (50)

  * Code (50):
    * accuracy >= 0.65: 50
    * accuracy >= 0.60: 45
    * accuracy >= 0.55: 40
    * accuracy >= 0.50: 35

  * Report (50):
    1. Introduction & Proposal (5)
    2. Methodology (20):
        a. Data processing (5)
        b. Feature generation (5)
        c. Network structure (5)
        d. Training & validation process (5)
    3. Evaluation & Results (20):
        a. Training & validation results (10)
        b. Performance comparing to your baselines (maybe different network structure) (5)
        c. Hyperparameter (learning rate, dropout, activation) (5)
    4. Conclusion (5)
  
   * Bonus (5):

       Add a screenshot of your test score (test accuracy) at the end of your report for the leaderboard bonus.
       5 bonus points for the top 3 on the leaderboard.

## Project Guidelines

#### Dataset Description
The data is stored in a dictionary, in which the key is the ID of a driver and the value is a list of his/her trajectories. For each trajectory, the basic element is similar to project 2. Each element in the trajectory is in the following format, [ plate, longitude, latitude, second_since_midnight, status, time ]. Data can be found at [Google Drive](https://drive.google.com/file/d/1jJky-XA2S-eBr51VgBHNLkdg4BV4l4mT/view?usp=sharing). The training data contain **400** drivers and **5**-day trajectories for each driver. We also provide a [validation dataset](https://drive.google.com/file/d/1jQcwvrL_p5Nw8bTC_gtPZQtvyV_Go8uj/view?usp=sharing), which contains 20 different drivers, and each of them also has 5-day trajectories. To successfully validate your model, you need to follow the same data pre-processing and generate the number of pairs of 100-step length sub-trajectories. Before your final submission, we can help test your model's performance, just email us with your code folder directly.
#### Feature Description 
* **Plate**: Plate means the taxi's plate. In this project, we change them to 0~500 to keep anonymity. The same plate means the same driver, so this is the target label for the classification. 
* **Longitude**: The longitude of the taxi.
* **Latitude**: The latitude of the taxi.
* **Second_since_midnight**: Seconds have passed since midnight.
* **Status**: 1 means the taxi is occupied and 0 means a vacant taxi.
* **Time**: Timestamp of the record.

#### Problem Definition
Given two 100-step length sub-trajectory trajectories,  you need to predict whether those two given trajectories belong to the same driver. 

#### Evaluation 
5 days of another 100 drivers' trajectories will be used to evaluate your submission. And test trajectories are not in the data/folder. You can evaluate the validation set based on the trajectories we provided. 

##### Feature Description of validation data
* **Longitude**: The longitude of the taxi.
* **Latitude**: The latitude of the taxi.
* **Second_since_midnight**: Seconds have passed since midnight.
* **Status**: 1 means the taxi is occupied and 0 means a vacant taxi.
* **Time**: Timestamp of the record.

#### Submission Guideline
Please compress all the below files into a zipped file and submit the zip file (firstName_lastName_P3.zip) to Canvas. 

* **Model Prediction**
    ```python
    def run(data, model):
        """
        
        Input:
            Data: the output of process_data function.
            Model: your model.
        Output:
            prediction: the predicted label(plate) of the data, an int value.
        
        """
        return prediction
  ```

## Tips of Using GPU on Turing Server

* Following the previous [instruction](https://github.com/UrbanIntelligence/CS586-DS504-Spring2024/blob/master/project2/Turing_Setup_Instructions.pdf), go to the folder where you train the model and activate the environment.

* Submit job on Turing server
   ```shell
   #!/bin/bash
   #SBATCH -A cs586
   #SBATCH -p academic
   #SBATCH -N 1
   #SBATCH -c 8
   #SBATCH --gres=gpu:1
   #SBATCH -t 12:00:00
   #SBATCH --mem 12G
   #SBATCH --job-name="p3"
    
   python main.py train
   ```


## Some Tips
Setup information could also be found in the [PDF](https://github.com/UrbanIntelligence/CS586-DS504-Spring2026/blob/main/project2/Turing_Setup_Instructions_2026S.pdf)
* Anaconda and virtual environment set tup
   * [Download and install anaconda](https://www.anaconda.com/distribution/)
   * [Create a virtual environment with commands](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
* Deep learning package
   * [Pytorch](https://pytorch.org/tutorials/)
* Open source GPU
   * [Turing](https://arc.wpi.edu/cluster-documentation/build/html/index.html)
   * [Kaggle](https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu)
* **Keywords**. 
   * If you are wondering where to start, you can try to search "sequence classification", "sequence to sequence" or "sequence embedding" in Google or Github, this might provide you some insights.

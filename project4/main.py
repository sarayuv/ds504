import argparse
import sys
from train import train_model
from test import test_model

def main():
    parser = argparse.ArgumentParser(
        description="Run training or testing for the taxi driver verification."
    )
    
    # This argument ("mode") must be either "train" or "test".
    parser.add_argument("mode", choices=["train", "test"], help="Choose 'train' or 'test'")


    parser.add_argument("--test_dir", default="./test_data",
                        help="Path to the folder containing test CSV files")

    args = parser.parse_args()

    # 2. Based on the mode, we either train or test.
    if args.mode == "train":
        print("Starting training process...")
        train_model()
    elif args.mode == "test":
        print("Starting testing process...")
        # Pass the test_dir argument to the test function, so it can find the CSV files.
        test_model(args.test_dir)
    else:
        print("Invalid mode. Choose 'train' or 'test'.")

if __name__ == '__main__':
    main()

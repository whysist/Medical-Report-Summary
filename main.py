# =============================
# main.py
# =============================
# Runs training, evaluation, and compression pipeline

from train import train_model
from evaluate import evaluate_model
from compress import compress_model

if __name__ == "__main__":
    # print("STEP 1: Training model...")
    # train_model()

    # print("\nSTEP 2: Evaluating model...")
    # evaluate_model()

    print("\nSTEP 3: Applying compression...")
    compress_model()




import pandas as pd
from drift_detection import detect_drift
import os

REFERENCE_PATH = os.path.join("data", "reference_data.csv")
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PRODUCTION_PATH = os.path.join(BASE_DIR, "data", "production_data.csv")

def monitor():

    reference = pd.read_csv(REFERENCE_PATH)

    production = pd.read_csv(PRODUCTION_PATH)

    reference = reference.drop(columns=["fraud"])
    production = production.drop(columns=["prediction"])

    report = detect_drift(reference, production)

    for feature, result in report.items():

        if result["drift_detected"]:
            print(f"DRIFT DETECTED in {feature}")
            return True

    print("No drift detected")
    return False


if __name__ == "__main__":

    drift = monitor()

    if drift:
        print("Trigger retraining pipeline")
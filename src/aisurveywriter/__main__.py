import os
import argparse
import re

from aisurveywriter import start_survey_pipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to YAML file containing the Survey's pipeline configuration")
    parser.add_argument("--credentials", type=str, default="credentials.yaml", help="YAML file containing your API keys")
    return parser.parse_args()

def main():
    args = parse_args()
    start_survey_pipeline(args.config, args.credentials)

if __name__ == "__main__":
    main()
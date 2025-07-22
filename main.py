import argparse
import logging
import sys
import yaml
from src.workflow import run_workflow

# sample: python3 main.py --config configs/Bi_PBE_spinless.yaml  

def main():
    parser = argparse.ArgumentParser(description="Map tight-binding parameters.")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file (YAML).')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    run_workflow(config)

if __name__ == "__main__":
    main()
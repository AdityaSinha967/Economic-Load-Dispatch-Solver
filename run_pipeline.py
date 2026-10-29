"""Run from the project root: python run_pipeline.py"""
from src.eld_ai.dataset import make_dataset
from src.eld_ai.train import train

if __name__ == "__main__":
    make_dataset()
    train()

"""Evaluate TCP using the official CARLA/Bench2Drive closed-loop protocol."""
from pathlib import Path
from vlastudio.benchmark.carla.evaluate_tcp import main

if __name__ == '__main__':
    main(str(Path(__file__).resolve().parent/'config/val.yaml'))

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from groq import Groq

import src.data_prep as dp
import src.dense_index as di
import src.baselines as bl
import src.ensemble as en
import src.evaluate as ev


def check_file_exists_and_valid(file_path: Path, min_size: int = 0) -> bool:
    # Check if file exists and meets minimum size requirement
    if not file_path.exists():
        return False
    if file_path.stat().st_size <= min_size:
        return False
    return True


def run_phase_1_data_prep():
    # Phase 1: Data preparation (checkpoints: passages.json, embeddings.npy)
    passages_file = Path("data/passages.json")
    embeddings_file = Path("data/embeddings.npy")
    
    if check_file_exists_and_valid(passages_file, min_size=1000) and check_file_exists_and_valid(embeddings_file, min_size=100000):
        print("Phase 1 (Data Prep): Skipped - outputs exist")
        return True
    
    print("Phase 1 (Data Prep): Running...")
    try:
        dp.main_data_prep()
        print("Phase 1 (Data Prep): Complete")
        return True
    except Exception as e:
        print(f"Phase 1 (Data Prep): Failed - {e}")
        return False


def run_phase_2_dense_index():
    # Phase 2: FAISS index construction (checkpoint: faiss.index)
    index_file = Path("data/faiss.index")
    
    if check_file_exists_and_valid(index_file, min_size=1000):
        print("Phase 2 (Dense Index): Skipped - output exists")
        return True
    
    print("Phase 2 (Dense Index): Running...")
    try:
        di.build_faiss_index()
        print("Phase 2 (Dense Index): Complete")
        return True
    except Exception as e:
        print(f"Phase 2 (Dense Index): Failed - {e}")
        return False


def run_phase_3_baselines(client):
    # Phase 3: Baseline experiments (checkpoint: baseline_results.json)
    results_file = Path("data/baseline_results.json")
    
    if check_file_exists_and_valid(results_file, min_size=100):
        print("Phase 3 (Baselines): Skipped - output exists")
        return True
    
    print("Phase 3 (Baselines): Running...")
    try:
        bl.run_all_baselines(
            'data/queries.json',
            'data/passages.json',
            'data/examples.json',
            client
        )
        print("Phase 3 (Baselines): Complete")
        return True
    except Exception as e:
        print(f"Phase 3 (Baselines): Failed - {e}")
        return False


def run_phase_4_ensemble(client):
    # Phase 4: REPLUG ensemble (checkpoint: ensemble_results.json)
    results_file = Path("data/ensemble_results.json")
    
    if check_file_exists_and_valid(results_file, min_size=100):
        print("Phase 4 (Ensemble): Skipped - output exists")
        return True
    
    print("Phase 4 (Ensemble): Running...")
    try:
        en.run_replug_ensemble(
            'data/queries.json',
            'data/passages.json',
            'data/faiss.index',
            'data/examples.json',
            client
        )
        print("Phase 4 (Ensemble): Complete")
        return True
    except Exception as e:
        print(f"Phase 4 (Ensemble): Failed - {e}")
        return False


def run_phase_5_evaluation():
    # Phase 5: Evaluation (checkpoint: results.json, k_variance.png)
    results_file = Path("data/results.json")
    plot_file = Path("data/k_variance.png")
    
    if check_file_exists_and_valid(results_file, min_size=100) and check_file_exists_and_valid(plot_file, min_size=10000):
        print("Phase 5 (Evaluation): Skipped - outputs exist")
        return True
    
    print("Phase 5 (Evaluation): Running...")
    try:
        ev.main_evaluation(
            'data/baseline_results.json',
            'data/ensemble_results.json',
            'data/queries.json'
        )
        print("Phase 5 (Evaluation): Complete")
        return True
    except Exception as e:
        print(f"Phase 5 (Evaluation): Failed - {e}")
        return False


def main():
    # Main pipeline orchestration with checkpoint/resume logic
    load_dotenv()
    
    # Check for Groq API key
    if not os.getenv('GROQ_API_KEY'):
        print("Error: GROQ_API_KEY not found in environment")
        print("Please set up your .env file with GROQ_API_KEY")
        sys.exit(1)
    
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    print("=" * 50)
    print("REPLUG QA Pipeline")
    print("=" * 50)
    
    # Run phases sequentially with checkpoint/resume
    phases = [
        ("Data Preparation", lambda: run_phase_1_data_prep()),
        ("FAISS Index", lambda: run_phase_2_dense_index()),
        ("Baselines", lambda: run_phase_3_baselines(client)),
        ("REPLUG Ensemble", lambda: run_phase_4_ensemble(client)),
        ("Evaluation", lambda: run_phase_5_evaluation())
    ]
    
    completed = 0
    failed = 0
    
    for phase_name, phase_func in phases:
        print(f"\n--- {phase_name} ---")
        success = phase_func()
        if success:
            completed += 1
        else:
            failed += 1
            print(f"Warning: {phase_name} failed, continuing...")
    
    print("\n" + "=" * 50)
    print(f"Pipeline Summary: {completed} phases completed, {failed} phases failed")
    print("=" * 50)
    
    if completed == len(phases):
        print("All phases completed successfully!")
        return 0
    else:
        print("Some phases failed - please check the errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

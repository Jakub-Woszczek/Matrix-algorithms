import argparse
import subprocess
from pathlib import Path

def run_lab(lab_name: str):
    lab_path = Path(__file__).parent / lab_name / "main.py"
    if not lab_path.exists():
        print(f"❌ No main.py found in '{lab_name}'")
        return
    print(f"▶️ Running {lab_name} ...\n")
    subprocess.run(["python3", str(lab_path)], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific lab scripts.")
    parser.add_argument("--lab1", action="store_true", help="Run main.py from lab1 directory")
    parser.add_argument("--lab2", action="store_true", help="Run main.py from lab2 directory")
    parser.add_argument("--lab3", action="store_true", help="Run main.py from lab3 directory")
    parser.add_argument("--lab4", action="store_true", help="Run main.py from lab4 directory")
    args = parser.parse_args()

    if args.lab1:
        run_lab("lab1")
    elif args.lab2:
        run_lab("lab2")
    elif args.lab3:
        run_lab("lab3")
    elif args.lab4:
        run_lab("lab4")
    else:
        print("ℹ️ No lab specified. Use --lab1, --lab2, --lab3, or --lab4.")


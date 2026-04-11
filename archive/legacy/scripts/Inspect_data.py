from pathlib import Path

data_dir = Path("data")

for file in data_dir.glob("*.txt"):
    print(f"\n--- {file.name} ---")
    try:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            print(f"Total lines: {len(lines)}")
            print("First 5 lines:")
            for line in lines[:5]:
                print(line.strip())
    except Exception as e:
        print(f"Could not read {file.name}: {e}")
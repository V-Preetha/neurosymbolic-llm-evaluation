import argparse,json
from pathlib import Path
from .core import write_json
def main():
    p=argparse.ArgumentParser();p.add_argument("--analysis",type=Path,required=True);a=p.parse_args();report=json.loads(a.analysis.read_text(encoding="utf-8"));out={"dataset":report["dataset"],"paired":report.get("paired",{}),"gains":report.get("gains",{})};target=a.analysis.parent/"statistics.json";write_json(target,out);print(json.dumps(out,indent=2));print(f"[Statistics] Saved {target}")
if __name__=="__main__":main()

import argparse,json
from .data import smoke_dataset
def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=["proofwriter","folio"],required=True);p.add_argument("--limit",type=int,default=5);a=p.parse_args();print(json.dumps(smoke_dataset(a.dataset,a.limit),indent=2))
if __name__=="__main__":main()

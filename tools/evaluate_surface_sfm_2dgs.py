"""Exact same raw-render RGB evaluator as the repaired OFF/COLMAP baseline."""
from pathlib import Path
import argparse
import evaluate_foreground_fix_2dgs as evaluator
evaluator.OUT=Path(__file__).resolve().parents[1]/'outputs/sfm_surface_init_20261014'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['HULL','GATE'],required=True)
    evaluator.evaluate(p.parse_args().method)

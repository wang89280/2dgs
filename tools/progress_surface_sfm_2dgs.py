from pathlib import Path
import progress_foreground_fix_2dgs as progress
progress.OUT=Path(__file__).resolve().parents[1]/'outputs/sfm_surface_init_20261014'
if __name__=='__main__':progress.main()

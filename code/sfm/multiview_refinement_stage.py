"""User-facing multiview refinement stage with training-only geometry guard.

The lower-level image_refinement_stage.optimize also supports the unguarded
network ablation used in the benchmark. The normal pipeline always uses this
guarded entry when --image-refinement multiview is selected.
"""
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python'))
from multiview_refinement import refine_observations
from multiview_consistency import gate
from image_refinement_stage import optimize
from model_io import save

def execute(sourceRoot,imagesPath,outputRoot,mode,logRoot):
    if mode!='multiview':raise ValueError(mode)
    source,images,output=map(Path,(sourceRoot,imagesPath,outputRoot))
    network=output.parent/(output.name+'_network_proposals')
    guarded=output.parent/(output.name+'_image_proposals')
    refine_observations(source/'sfm',images,network)
    gate(source,network,guarded)
    result=optimize(source,output,guarded,'localization',logRoot,multiview=True)
    for path in (output/'image_refinement.json',output/'sfm/summary.json'):
        info=json.loads(path.read_text(encoding='utf8'))
        info['mode' if path.name=='image_refinement.json' else 'image_refinement_mode']='multiview'
        info.update(image_proposal_method='densesfm_multiview_with_training_geometry_guard',
                    image_ba_weighting='uniform',network_proposals=str(network),guarded_proposals=str(guarded),
                    independent_geometric_accuracy_claimed=False)
        save(path,info)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','images','output','logs'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();execute(a.source,a.images,a.output,'multiview',a.logs)

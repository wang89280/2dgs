import argparse,csv,json,sys
from pathlib import Path
from common import ROOT,save,command
from track_lifecycle import complete_graph,execute
from model_io import patch_xml

def backend(out,threads,initial_pairs,track_recovery=True,image_refinement='off',point_refinement='off',track_partition='off'):
    if track_partition not in ('off','view-consensus'):raise ValueError('Unknown track partition mode')
    if point_refinement not in ('off','robust'):raise ValueError('Unknown point refinement mode')
    if image_refinement not in ('off','localization','weighted','learned','multiview'):
        raise ValueError('Unknown image refinement mode: '+str(image_refinement))
    if image_refinement=='multiview':
        from multiview_refinement_stage import execute as refine_images
    elif image_refinement!='off':
        from image_refinement_stage import execute as refine_images
    images=json.loads((out/'shared/image_manifest.json').read_text(encoding='utf8'))
    with (out/'shared/observations.csv').open(newline='',encoding='utf8') as stream:
        observed={int(row['image']) for row in csv.DictReader(stream)}
    missing=set(range(len(images)))-observed
    if missing:
        raise ValueError('No surviving matched tracks for input images: '+', '.join(images[i]['name'] for i in sorted(missing)))
    if observed!=set(range(len(images))):raise ValueError('Observation image indices do not match the prepared images')
    base=out/'_work/initialization';base.mkdir(parents=True)
    command(out,'initialize',[ROOT/'bin/initialize_cameras.exe',out/'shared/images',out/'shared/observations.csv',base/'initial',str(initial_pairs)],threads)
    command(out,'incremental_sfm',[ROOT/'bin/incremental_sfm.exe',base/'initial',base/'sfm'],threads)
    patch_xml(base/'sfm')
    expected=(out/'shared/calibration_fixed.bin').read_bytes()
    if expected!=(base/'sfm/calibration.bin').read_bytes():raise RuntimeError('Fixed calibration changed')
    complete_graph(out/'shared',out/'_work/graph')
    save(out/'status.json',dict(stage='track_reconstruction'))
    execute(base,out/'_work/graph',out/'_work/track_reconstruction',engine='rebuild_tracks',track_recovery=track_recovery)
    save(out/'status.json',dict(stage='track_maintenance'))
    geometry=out/'model' if image_refinement=='off' and point_refinement=='off' and track_partition=='off' else out/'_work/geometry_before_refinement'
    execute(out/'_work/track_reconstruction',out/'_work/graph',geometry,engine='maintain_tracks',track_recovery=track_recovery)
    if expected!=(geometry/'sfm/calibration.bin').read_bytes():raise RuntimeError('Track processing changed fixed calibration')
    if image_refinement!='off':
        save(out/'status.json',dict(stage='image_refinement',mode=image_refinement))
        image_output=out/'model' if point_refinement=='off' and track_partition=='off' else out/'_work/image_refined'
        refined=Path(refine_images(geometry,out/'shared/images',image_output,image_refinement,out/'logs'))
        if refined.resolve()!=image_output.resolve():raise RuntimeError('Image refinement returned an unexpected model directory: '+str(refined))
        if expected!=(refined/'sfm/calibration.bin').read_bytes():raise RuntimeError('Image refinement changed fixed calibration')
        geometry=refined
    if point_refinement=='robust':
        from point_refinement_stage import refine as refine_points
        save(out/'status.json',dict(stage='point_refinement',mode=point_refinement))
        point_output=out/'model' if track_partition=='off' else out/'_work/point_refined'
        refine_points(geometry,point_output)
        geometry=point_output
    if track_partition=='view-consensus':
        from view_consensus_stage import run as partition_tracks
        save(out/'status.json',dict(stage='view_consensus_partition'))
        partition_tracks(geometry,out/'model')
    save(out/'model/sparse_complete.json',dict(complete=True,algorithm='final_improved',fixed_intrinsics=True,
        track_recovery=track_recovery,image_refinement=image_refinement,point_refinement=point_refinement,track_partition=track_partition,no_metashape_runtime=True,no_colmap_mapper=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--initial-pairs',type=int,default=200)
    p.add_argument('--track-recovery',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--image-refinement',choices=['off','localization','weighted','learned','multiview'],default='off')
    p.add_argument('--point-refinement',choices=['off','robust'],default='off')
    p.add_argument('--track-partition',choices=['off','view-consensus'],default='off')
    a=p.parse_args();backend(a.output,a.threads,a.initial_pairs,a.track_recovery,a.image_refinement,a.point_refinement,a.track_partition)

"""Final four-method comparison with an exact point-count/order/color control."""
import csv,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sfm_surface_init_20261014'
OLD=ROOT/'outputs/foreground_fix_2dgs_2-1_20261014';METHODS=['COLMAP','OFF','GATE','HULL']
KEYS=['psnr_db','psnr_nonblack_db','ssim','lpips_vgg_crop']
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def base(m):return OLD if m in ['OFF','COLMAP'] else OUT
def writecsv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sheet(names,look,page):
    cw,ch=320,290;canvas=Image.new('RGB',(5*cw,40+len(names)*ch),'#eeeeee');draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',13)
    except OSError:font=ImageFont.load_default()
    for j,name in enumerate(['GT']+METHODS):draw.text((j*cw+8,12),name,fill='black',font=font)
    for i,name in enumerate(names):
        box=look['HULL'][name]['crop_box_xyxy'];paths=[OUT/'results/HULL/test/gt'/f'{name}.png']+[base(m)/'results'/m/'test/renders'/f'{name}.png' for m in METHODS]
        for j,path in enumerate(paths):
            im=Image.open(path).convert('RGB').crop(box);im.thumbnail((cw-10,ch-42),Image.Resampling.LANCZOS)
            canvas.paste(im,(j*cw+(cw-im.width)//2,40+i*ch+30))
            label=name if j==0 else f"{name} fgPSNR {look[METHODS[j-1]][name]['psnr_nonblack_db']:.3f}"
            draw.text((j*cw+7,40+i*ch+7),label,fill='black',font=font)
    canvas.save(OUT/f'comparison_views_{page:02d}.png')
def main():
    assert read(OUT/'audit.json')['passed'];p=read(OUT/'protocol.json');s=read(OUT/'prepared.json')['stats'];g=read(OUT/'native_geometry_comparison.json')
    data={m:read(base(m)/'results'/m/'metrics.json') for m in METHODS}
    look={m:{r['name']:r for r in data[m]['rows'] if r['split']=='test'} for m in METHODS};names=p['test_names']
    assert all(list(look[m])==names for m in METHODS)
    for n in names:assert len({tuple(look[m][n]['crop_box_xyxy']) for m in METHODS})==1
    rows=[dict(method=m,**{k:data[m]['means']['test'][k] for k in KEYS},gaussians=data[m]['gaussians']) for m in METHODS]
    deltas=[dict(comparison='HULL_minus_'+m,**{k:data['HULL']['means']['test'][k]-data[m]['means']['test'][k] for k in KEYS},
        improved_nonblack_views=sum(look['HULL'][n]['psnr_nonblack_db']>look[m][n]['psnr_nonblack_db'] for n in names)) for m in ['GATE','OFF','COLMAP']]
    per=[dict(method=m,view=n,**{k:look[m][n][k] for k in KEYS},background_alpha=look[m][n]['alpha_bg_mean'],foreground_visible_fraction=look[m][n]['foreground_visible_fraction']) for m in METHODS for n in names]
    writecsv('comparison.csv',rows);writecsv('deltas.csv',deltas);writecsv('per_view.csv',per)
    for page in [1,2]:sheet(names[(page-1)*5:page*5],look,page)
    off=next(d for d in deltas if d['comparison']=='HULL_minus_OFF');gate=next(d for d in deltas if d['comparison']=='HULL_minus_GATE')
    favorable=off['psnr_nonblack_db']>0 and off['lpips_vgg_crop']<0 and gate['psnr_nonblack_db']>0 and gate['lpips_vgg_crop']<0
    decision='本次单场景结果同时支持相对OFF与同点数GATE的非黑PSNR/LPIPS改善；尚未证明跨场景稳定收益。' if favorable else '本次没有在非黑PSNR和LPIPS上同时超过OFF及同点数GATE；不将候选提升为默认。'
    lines=['轮廓约束SfM三角化 → 2DGS：2-1 完整对照','',decision,'',
        'ALIKED-N32 + LightGlue、原始特征坐标、轨迹归属、相机内外参、图像、2DGS源码与参数均保持不变。本模块利用70个训练视角的已有前景mask，在原始观测JᵀJ定义的度量下修正三维点，使其满足多视图轮廓约束；不使用训练好的2DGS代理，也不读取10张评估RGB生成约束。','',
        'HULL与GATE各新训练一次30,000步，seed=0，70训练/10评估、1600px。OFF与COLMAP复用相同修复版2DGS的已核验旧结果。GATE保留与HULL完全相同的点编号、点数、顺序和颜色，只恢复原始XYZ，因此HULL−GATE隔离点位置修正的作用。严格原点过滤FILTER只保留为几何诊断，未另训练。','',
        f"原始2DGS种子点 {s['input_points']}；严格不修正过滤保留 {s['filter_points']}；HULL/GATE均为 {s['hull_points']}。其中 {s['recovered_by_constrained_triangulation']} 个点被修正，{s['rejected_initialization_points']} 个点未用于本次初始化。",'',
        '固定原始观测的SfM对照（3072×2048畸变图坐标，40,258轨迹、257,967观测；没有删除难点重报RMSE）：','',
        f"- 原始RMSE：{g['original']['rmse_px']:.8f}px；候选：{g['candidate_all_original_observations']['rmse_px']:.8f}px。",'',
        '加入轮廓约束牺牲了一部分原始特征拟合；轮廓可行性是本方法直接优化的约束，不能把它当独立几何精度或用筛选后的低RMSE宣称SfM涨点。相机精度在本轮没有改变。','',
        '| 方法 | 全图PSNR ↑ | 非黑PSNR ↑ | SSIM ↑ | 物体框LPIPS ↓ | 高斯数 |','|---|---:|---:|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['method']} | {row['psnr_db']:.6f} | {row['psnr_nonblack_db']:.6f} | {row['ssim']:.8f} | {row['lpips_vgg_crop']:.8f} | {row['gaussians']} |")
    lines+=['','| HULL减对照 | 全图PSNR差 | 非黑PSNR差 | SSIM差 | 框LPIPS差 | 非黑PSNR提高视角 |','|---|---:|---:|---:|---:|---:|']
    for row in deltas:lines.append(f"| {row['comparison']} | {row['psnr_db']:+.6f} | {row['psnr_nonblack_db']:+.6f} | {row['ssim']:+.8f} | {row['lpips_vgg_crop']:+.8f} | {row['improved_nonblack_views']}/10 |")
    lines+=['','最差视角与背景检查：','']
    for m in METHODS:
        worst=min(look[m].values(),key=lambda r:r['psnr_nonblack_db'])
        lines.append(f"- {m}：最差 {worst['name']}，非黑PSNR {worst['psnr_nonblack_db']:.4f}dB；平均背景alpha {data[m]['means']['test']['alpha_bg_mean']:.7f}。")
    review=OUT/'visual_review.md';lines+=['','实际视觉检查：','',review.read_text(encoding='utf8') if review.exists() else '待实际检查全部10个视角，不预填视觉结论。','',
        '结论边界：这是一种SfM后端三角化与种子点导出改动，下游2DGS优化代码未改。当前只有一个场景和一个种子，而且同一评估场景此前已多次使用，不能称为盲测或外部泛化验证。初始SfM使用全量80图，属于给定全量SfM几何的RGB评估视角对照；无独立3D真值，不能声称毫米精度或99%接近原厂。点约束和可行性改善不等价于真实表面误差下降。原始文件、原始候选及全部失败结果保留。', '',
        '思路参考：[Initialize to Generalize, 2025 preprint](https://arxiv.org/abs/2510.17479)及[作者代码](https://github.com/zss171999645/ItG-GS)支持研究初始化质量与多视图一致性的方向；本轮没有实现该文的频率图像增强或GS自初始化。协方差度量下的轮廓修正是此处独立实现的工程候选，未证明研究新颖性，也不能把论文收益移植为本实验收益。']
    for page in [1,2]:
        path=str(OUT/f'comparison_views_{page:02d}.png').replace('\\','/').replace('/mnt/f/','F:/')
        lines.append(f'\n[评估视角对照图 {page}](<{path}>)')
    report=OUT/'轮廓约束SfM与2DGS报告.md';report.write_text('\n'.join(lines),encoding='utf8')
    (OUT/'report_status.json').write_text(json.dumps(dict(metrics_complete=True,visual_review_complete=review.exists(),favorable_on_primary_metrics=favorable,report=str(report)),indent=2),encoding='utf8')
    print(json.dumps(dict(comparison=rows,deltas=deltas),indent=2))
if __name__=='__main__':main()

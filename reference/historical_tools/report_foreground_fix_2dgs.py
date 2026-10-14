"""Summarize all three fixed runs and paired changes from the untouched originals."""
import csv,json,statistics
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/foreground_fix_2dgs_2-1_20261014';OLD=ROOT/'outputs/latest_2dgs_2-1_20261014'
METHODS=['COLMAP','OFF','ON'];KEYS=['psnr_db','psnr_nonblack_db','ssim','ssim_crop','lpips_vgg_crop']
def read(p):return json.loads(p.read_text(encoding='utf8'))
def writecsv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def win(p):return str(p).replace('/mnt/f/','F:/').replace('\\','/')

def sheet(names,look,filename,method=None):
    labels=['GT','COLMAP fixed','OFF fixed','ON fixed'] if method is None else ['GT',method+' original',method+' fixed']
    cw,ch=330,295;canvas=Image.new('RGB',(cw*len(labels),40+ch*len(names)),'#eeeeee');draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:font=ImageFont.load_default()
    for j,label in enumerate(labels):draw.text((j*cw+8,12),label,fill='black',font=font)
    for i,name in enumerate(names):
        box=look['ON'][name]['crop_box_xyxy']
        paths=[OUT/'results/ON/test/gt'/f'{name}.png']
        if method is None:paths += [OUT/'results'/m/'test/renders'/f'{name}.png' for m in METHODS]
        else:paths += [OLD/'results'/method/'test/renders'/f'{name}.png',OUT/'results'/method/'test/renders'/f'{name}.png']
        for j,path in enumerate(paths):
            im=Image.open(path).convert('RGB').crop(box);im.thumbnail((cw-12,ch-35),Image.Resampling.LANCZOS)
            canvas.paste(im,(j*cw+(cw-im.width)//2,40+i*ch+27));draw.text((j*cw+8,40+i*ch+6),name,fill='black',font=font)
    canvas.save(OUT/filename)

def main():
    assert read(OUT/'audit.json')['passed'];p=read(OUT/'protocol.json')
    new={m:read(OUT/'results'/m/'metrics.json') for m in METHODS};old={m:read(OLD/'results'/m/'metrics.json') for m in METHODS}
    look={m:{r['name']:r for r in new[m]['rows'] if r['split']=='test'} for m in METHODS}
    names=p['test_names'];assert all(list(look[m])==names for m in METHODS)
    for name in names:assert len({tuple(look[m][name]['crop_box_xyxy']) for m in METHODS})==1
    rows=[];delta=[];per=[]
    for m in METHODS:
        for label,data in [('original',old[m]),('fixed',new[m])]:
            rr=[r for r in data['rows'] if r['split']=='test']
            rows.append(dict(method=m,training=label,**{k:data['means']['test'][k] for k in KEYS},
                worst_nonblack_psnr=min(r['psnr_nonblack_db'] for r in rr),gaussians=data['gaussians']))
        delta.append(dict(comparison=m+'_fixed_minus_original',**{k:new[m]['means']['test'][k]-old[m]['means']['test'][k] for k in KEYS}))
        oldlook={r['name']:r for r in old[m]['rows'] if r['split']=='test'}
        for n in names:per.append(dict(method=m,view=n,**{k:look[m][n][k] for k in KEYS},alpha_bg=look[m][n]['alpha_bg_mean'],
            visible_fraction=look[m][n]['foreground_visible_fraction'],nonblack_delta=look[m][n]['psnr_nonblack_db']-oldlook[n]['psnr_nonblack_db']))
    for other in ['OFF','COLMAP']:
        delta.append(dict(comparison='fixed_ON_minus_'+other,**{k:new['ON']['means']['test'][k]-new[other]['means']['test'][k] for k in KEYS}))
    writecsv('comparison.csv',rows);writecsv('deltas.csv',delta);writecsv('per_view.csv',per)
    for page in [0,1]:
        sheet(names[page*5:page*5+5],look,f'comparison_views_{page+1:02d}.png')
        for m in METHODS:sheet(names[page*5:page*5+5],look,f'before_after_{m}_{page+1:02d}.png',m)
    lines=['2DGS 前景稳定性修复：2-1 三方法对照','',
        'COLMAP、OFF、ON 分别使用各自原有 ALIKED+LightGlue SfM 初始化，统一运行修复后的 2DGS 各一次 30,000 步，seed=0、1600px、70 训练/10 评估视角。训练图和稀疏模型保持原始字节；只从 70 张训练图生成掩膜，没有生成或读取评估 mask。旧 30,000 步模型保留。','',
        '本轮共同修复：显式前景/背景 alpha 损失；只在可靠前景约束单位法线，15k–20k 逐渐启用；保留并正确使用屏幕半径统计；训练后半程继续异常裁剪。原 RGB 损失、渲染器和 CUDA 扩展保持不变。没有对最终渲染图做掩膜后处理。','',
        '| 方法 | 训练版本 | 全图 PSNR ↑ | 非黑区域 PSNR ↑ | SSIM ↑ | 框 LPIPS ↓ | 最差视角非黑 PSNR ↑ | 高斯数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['method']} | {r['training']} | {r['psnr_db']:.4f} | {r['psnr_nonblack_db']:.4f} | {r['ssim']:.6f} | {r['lpips_vgg_crop']:.6f} | {r['worst_nonblack_psnr']:.4f} | {r['gaussians']} |")
    lines+=['','| 对照（前者减后者） | 全图 PSNR 差 | 非黑 PSNR 差 | SSIM 差 | 框 LPIPS 差 |','|---|---:|---:|---:|---:|']
    for r in delta:lines.append(f"| {r['comparison']} | {r['psnr_db']:+.4f} | {r['psnr_nonblack_db']:+.4f} | {r['ssim']:+.6f} | {r['lpips_vgg_crop']:+.6f} |")
    lines+=['','PSNR/SSIM 越高越好；LPIPS 越低越好。非黑区域仍按 GT max RGB >0.05，物体框仍外扩16px，与旧评估一致。三方法使用相同评估视角和裁剪框。','',
        '背景平均 alpha（亮度阈值的背景补集，包含少量轮廓边界，并非严格语义背景）：','']
    for m in METHODS:lines.append(f"- {m}：{new[m]['means']['test']['alpha_bg_mean']:.6f}；物体区可见比例 {new[m]['means']['test']['foreground_visible_fraction']:.6f}。")
    review=OUT/'visual_review.md';lines+=['','实际图像检查：','',review.read_text(encoding='utf8') if review.exists() else '尚待逐视角实际查看；不预填视觉结论。','']
    for page in [1,2]:lines.append(f'[三方法评估视角 {page}](<{win(OUT/f"comparison_views_{page:02d}.png")}>)')
    lines+=['','评估限制：单场景、单种子、联合修复包试验，不能分离各修复的独立贡献。所有方法的 SfM 几何使用全量 80 张图，因此属于给定全量 SfM 几何的 2DGS RGB 评估视角对照；没有独立三维真值，不能据此声称毫米精度或 99% 接近 Metashape。超参数在训练前冻结，仅以最终30k报告，不按评估图挑检查点。','',
        '原始文件和代码的哈希、训练 mask 清单、修改源码哈希、冻结参数见 protocol.json；最终完整性检查见 audit.json。当前训练修复属于工程纠错，修复涨幅不能直接作为 SfM 新模块收益。']
    report=OUT/'修复后2DGS对比报告.md';report.write_text('\n'.join(lines),encoding='utf8')
    (OUT/'report_status.json').write_text(json.dumps(dict(metrics_complete=True,visual_review_complete=review.exists(),report=win(report)),indent=2),encoding='utf8')
    print(json.dumps(dict(comparison=rows,deltas=delta),ensure_ascii=False))

if __name__=='__main__':main()

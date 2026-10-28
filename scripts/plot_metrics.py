"""Plot final matrices and paired per-image distributions without checkpoints."""
from pathlib import Path
import argparse
import json
import math
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from summarize_results import load_json, sha256_file, summarize_dataset


def positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError('Value must be finite and positive')
    return parsed


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('Value must be positive')
    return parsed


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Formal evaluation directory')
    p.add_argument('--output', type=Path, required=True, help='New figure directory')
    p.add_argument('--dpi', type=positive_int, default=180)
    # Edit these or supply CLI values to tune publication typography.
    p.add_argument('--axis-size', type=positive_float, default=54)
    p.add_argument('--tick-size', type=positive_float, default=48)
    p.add_argument('--cell-size', type=positive_float, default=32)
    p.add_argument('--annotation-size', type=positive_float, default=36)
    p.add_argument('--font-family', default='Times New Roman')
    p.add_argument('--strict-font', action='store_true', help='Fail if the requested font is unavailable')
    return p


def compute_plot_statistics(mse, ssim, lpips):
    """Pool images for matrix cells and pool states within each distribution image.

    RMSE is a square root after the relevant MSE pooling; individual RMSEs must
    not be averaged. A violin observation is one image, not one state pair.
    """
    arrays = [np.asarray(value, dtype=np.float64) for value in (mse, ssim, lpips)]
    shape = arrays[0].shape
    if len(shape) != 3 or shape[0] == 0 or shape[1:] != (10, 10):
        raise ValueError('Metric arrays must have non-empty shape (N,10,10)')
    if any(value.shape != shape or not np.isfinite(value).all() for value in arrays):
        raise ValueError('Metric arrays must have identical shape and finite values')
    if (arrays[0] < 0).any():
        raise ValueError('MSE must be non-negative')
    mask = ~np.eye(10, dtype=bool)
    result = {}
    for name, values in zip(('MSE', 'SSIM', 'LPIPS'), arrays):
        result[name] = {
            'matrix': values.mean(axis=0, dtype=np.float64),
            'correct_per_image': np.diagonal(values, axis1=1, axis2=2).mean(axis=1, dtype=np.float64),
            'wrong_per_image': values[:, mask].mean(axis=1, dtype=np.float64),
        }
    result['RMSE'] = {name: np.sqrt(value) for name, value in result['MSE'].items()}
    return result


def resolve_font(requested, *, strict=False):
    available = {font.name for font in font_manager.fontManager.ttflist}
    if requested in available:
        return requested
    if strict:
        raise ValueError(f'Requested font is unavailable: {requested}')
    warnings.warn(f'{requested} unavailable; using DejaVu Serif. The fallback is recorded.', stacklevel=2)
    return 'DejaVu Serif'


def matrix_title(metric, dataset):
    if metric == 'LPIPS':
        return 'LPIPS-Alex perceptual distance'
    if metric == 'SSIM':
        return 'SSIM v2 in native image space'
    domain = 'grayscale' if dataset == 'mnist' else 'RGB'
    return f'{metric} in native [0,1] {domain}'


def main():
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    summary = load_json(args.input / 'reconstruction_summary.json')
    manifest = load_json(args.input / 'evaluation_manifest.json')
    verified = summarize_dataset(args.input, summary['dataset'])
    with np.load(args.input / 'per_image_metrics.npz', allow_pickle=False) as z:
        mse, ssim, lpips = (z['seeded_cross_' + k].astype(np.float64) for k in ('mse','ssim','lpips'))
    statistics = compute_plot_statistics(mse, ssim, lpips)
    labels = [f'{nm}/{10 * power}' for nm in (405,640) for power in (0,5,10,15,20)]
    family = resolve_font(args.font_family, strict=args.strict_font)
    plt.rcParams.update({'font.family':family,'font.weight':'bold','axes.labelweight':'bold',
                         'axes.titleweight':'bold','axes.linewidth':3.0,'pdf.fonttype':42})
    args.output.mkdir(parents=True)
    metrics = [('MSE','magma'),('RMSE','magma'),('SSIM','viridis'),('LPIPS','magma')]
    fig, axes = plt.subplots(2,2,figsize=(38,34),layout='constrained')
    for ax,(name,cmap) in zip(axes.ravel(),metrics):
        matrix = statistics[name]['matrix']
        im=ax.imshow(matrix,cmap=cmap,interpolation='nearest')
        ax.set_xticks(range(10),labels,rotation=45,ha='right',fontsize=args.tick_size)
        ax.set_yticks(range(10),labels,fontsize=args.tick_size)
        ax.set_xlabel('Receiver (nm / mJ, 10 s)',fontsize=args.axis_size)
        ax.set_ylabel('Sender (nm / mJ, 10 s)',fontsize=args.axis_size)
        ax.set_title(matrix_title(name, summary['dataset']),fontsize=args.axis_size,pad=24)
        for i in range(10):
            for j in range(10):
                rgba=im.cmap(im.norm(matrix[i,j])); lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j,i,f'{matrix[i,j]:.4f}' if name=='MSE' else f'{matrix[i,j]:.3f}',
                        color='black' if lum>.53 else 'white',ha='center',va='center',
                        fontsize=args.cell_size,fontweight='bold')
        for i in range(10):
            ax.add_patch(plt.Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor='#00bb77',lw=2.5))
        cbar=fig.colorbar(im,ax=ax,fraction=.045,pad=.035)
        cbar.ax.tick_params(labelsize=args.tick_size,width=2.5,length=8)
        cbar.set_label('Higher is better' if name=='SSIM' else 'Lower is better',
                       fontsize=args.annotation_size,fontweight='bold')
        cbar.outline.set_linewidth(3)
        ax.tick_params(width=2.5,length=8)
    for ext in ('png','pdf'):
        fig.savefig(args.output/f'{summary["dataset"]}_02_state_mismatch_10x10_metrics.{ext}',
                    dpi=args.dpi,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(38,11),layout='constrained')
    for ax,(name,_) in zip(axes,metrics):
        correct=statistics[name]['correct_per_image']
        wrong=statistics[name]['wrong_per_image']
        datasets=[correct,wrong]
        vp=ax.violinplot(datasets,positions=[1,2],showextrema=False)
        for body,color in zip(vp['bodies'],['#58b2d2','#e28a3b']):
            body.set_facecolor(color); body.set_edgecolor('black'); body.set_alpha(.85)
        ax.boxplot(datasets,positions=[1,2],widths=.2,showfliers=False,
                   medianprops={'color':'black','linewidth':3},
                   boxprops={'linewidth':2},whiskerprops={'linewidth':2},capprops={'linewidth':2})
        ax.set_xticks([1,2],['Matched\n10 states','Mismatched\n90 pairs'],fontsize=args.annotation_size+2)
        ax.tick_params(axis='y',labelsize=args.tick_size,width=2.5,length=8)
        ax.set_ylabel(name,fontsize=args.axis_size)
        ax.set_title(f'Per-image {name}',fontsize=args.axis_size,pad=24)
        ax.grid(axis='y',alpha=.2)
    fig.supxlabel('One observation per test image; state-mean metrics (RMSE = sqrt of state-mean MSE).',
                  fontsize=args.annotation_size)
    for ext in ('png','pdf'):
        fig.savefig(args.output/f'{summary["dataset"]}_03_correct_wrong_metric_distributions.{ext}',
                    dpi=args.dpi,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)
    record={'dataset':summary['dataset'],'ssim':manifest['metrics']['ssim'],
            'font_family':family,'count':mse.shape[0],
            'schema':'oshc-vqvae-metric-figure-specification-v2',
            'inputs_sha256':verified['source_sha256'],
            'plot_script_sha256':sha256_file(Path(__file__)),
            'evaluation_summary_verified':verified['summary_verified'],
            'runtime':{'numpy':np.__version__,'matplotlib':matplotlib.__version__},
            'distribution_unit':'one image; mean across 10 correct or 90 wrong conditions',
            'matrix_reduction':'mean over all images for each ordered sender/receiver cell',
            'matrix_axes':{'rows':'sender record','columns':'receiver record'},
            'rmse_matrix':'sqrt of pooled per-cell MSE; never mean of per-image RMSE',
            'rmse_distribution':'sqrt of each image condition-mean MSE',
            'lpips_input_policy':manifest['metrics']['lpips'],
            'labels_note':'Q10_mJ = 10 * power_mw; 0 mJ rows are separate sweep references',
            'state_display_units':'pump nm / Q10 mJ at 10 s',
            'style':vars(args)}
    record['outputs_sha256']={path.name:sha256_file(path) for path in sorted(args.output.iterdir()) if path.is_file()}
    (args.output/'figure_specification.json').write_text(json.dumps(record,indent=2,default=str)+'\n',encoding='utf-8')
    print(args.output)


if __name__=='__main__':
    main()

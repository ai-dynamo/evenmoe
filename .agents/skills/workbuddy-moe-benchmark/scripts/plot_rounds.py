# SPDX-License-Identifier: Apache-2.0
"""Plot every measured formal round, colored by server launch; requires matplotlib."""
import argparse
import json
from pathlib import Path


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output.mkdir(parents=True, exist_ok=False)
    records = [r for r in summary['round_records'] if r['phase'] == 'formal']
    manifest = []
    for index, (policy, seed) in enumerate(sorted({(r['policy'], r['seed']) for r in records})):
        rows = [r for r in records if (r['policy'], r['seed']) == (policy, seed)]
        concurrencies = sorted({r['concurrency'] for r in rows})
        sessions = sorted({r['session'] for r in rows})
        colors = {s: plt.get_cmap('tab10')(i % 10) for i, s in enumerate(sessions)}
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        plotted = []
        rounds = sorted({r['round'] for r in rows})
        markers = {r: ['o', 's', '^', 'D', 'v', 'P'][i % 6] for i, r in enumerate(rounds)}
        for x, concurrency in enumerate(concurrencies):
            group = [r for r in rows if r['concurrency'] == concurrency]
            valid = [r['ktokens_s'] for r in group if r['included']]
            if valid:
                axes[0].boxplot([valid], positions=[x], widths=0.55, showfliers=False)
            for j, r in enumerate(group):
                value = r.get('ktokens_s')
                if value is None:
                    continue
                jitter = 0.32 * ((j + 0.5) / len(group) - 0.5)
                marker = markers[r['round']] if r['included'] else 'x'
                axes[0].scatter(x+jitter, value, color=colors[r['session']], marker=marker, zorder=3)
                plotted.append(r)
            for session in sessions:
                trajectory = sorted([r for r in group if r['session'] == session and r['included']],
                                    key=lambda r: r['round'])
                offset = 0.04 * (sessions.index(session) - (len(sessions)-1)/2)
                axes[1].plot([r['round'] + offset for r in trajectory], [r['ktokens_s'] for r in trajectory],
                             marker='o', color=colors[session],
                             linestyle=['-', '--', ':', '-.'][x % 4],
                             label=f'C{concurrency} / S{session}')
        axes[0].set_xticks(range(len(concurrencies)), [str(c) for c in concurrencies])
        axes[0].set_xlabel('Configured global concurrency')
        axes[1].set_xlabel('Formal round (small offsets distinguish launches)')
        axes[1].set_xticks(rounds)
        for ax in axes:
            ax.set_ylabel(summary['units'])
            ax.grid(axis='y', alpha=0.2)
        axes[0].legend(handles=[Line2D([], [], marker='o', linestyle='', color=colors[s],
                                     label=f'Server launch {s}') for s in sessions]
                       + [Line2D([], [], marker=markers[r], linestyle='', color='gray', label=f'Round {r}')
                          for r in rounds]
                       + [Line2D([], [], marker='x', linestyle='', color='black', label='Excluded')],
                       fontsize=8)
        axes[1].legend(fontsize=7)
        no_value = sum(r.get('ktokens_s') is None for r in rows)
        fig.suptitle(f'{policy}, seed {seed}: {len(plotted)} measured points; '
                     f'{no_value} attempts without throughput\n'
                     'Boxes: included rounds; points: all measured attempts. Rounds are not independent launches.')
        fig.tight_layout()
        name = f'rounds-{index:03d}'
        for suffix in ['svg', 'png']:
            fig.savefig(output/f'{name}.{suffix}', dpi=160)
        plt.close(fig)
        manifest.append({'policy': policy, 'seed': seed, 'figure': name,
                         'plotted_records': plotted,
                         'unplottable_records': [r for r in rows if r.get('ktokens_s') is None]})
    (output/'plot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text()), args.output_dir)


if __name__ == '__main__':
    main()

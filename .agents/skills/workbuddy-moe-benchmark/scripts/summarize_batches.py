# SPDX-License-Identifier: Apache-2.0
"""Summarize rank-local execution batches; never sum TP-replicated counters."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values)-1)*fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi]-values[lo])*(position-lo)


def summarize_batches(records, coverage):
    groups = defaultdict(list)
    seen = set()
    for row in records:
        key = tuple(row[k] for k in ('policy', 'concurrency', 'seed', 'session', 'round',
                                    'attempt', 'rank', 'execution_phase'))
        identity = (*key, row['step_id'])
        if identity in seen:
            raise ValueError(f'Duplicate rank-local step: {identity}')
        seen.add(identity)
        for field in ('requests', 'tokens', 'padded_tokens'):
            n = row[field]
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ValueError(f'Invalid {field}')
        if row['padded_tokens'] < row['tokens']:
            raise ValueError('Padded tokens cannot be smaller than valid tokens')
        for field in ('tp_rank', 'dp_rank', 'ep_rank', 'timestamp_ns', 'clock_domain'):
            if field not in row:
                raise ValueError(f'Missing topology/time field: {field}')
        groups[key].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        topology = {(r['tp_rank'], r['dp_rank'], r['ep_rank'], r['clock_domain']) for r in rows}
        if len(topology) != 1:
            raise ValueError('Rank topology or clock domain changed within a group')
        stats = {}
        for field in ('requests', 'tokens', 'padded_tokens'):
            values = [r[field] for r in rows]
            stats[field] = {'min': min(values), 'p10': quantile(values, .1),
                            'p50': quantile(values, .5), 'p90': quantile(values, .9),
                            'max': max(values), 'mean': sum(values)/len(values)}
        result.append(dict(zip(('policy', 'concurrency', 'seed', 'session', 'round',
                                'attempt', 'rank', 'execution_phase'), key),
                           steps=len(rows), distributions=stats,
                           topology=dict(zip(('tp_rank', 'dp_rank', 'ep_rank', 'clock_domain'),
                                             next(iter(topology))))))
    return {'coverage': coverage, 'coverage_verified': False,
            'weighting': 'one observation per rank-local execution step; ranks never summed',
            'groups': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=Path, required=True)
    parser.add_argument('--coverage', choices=['all_steps', 'sampled', 'diagnostic_only'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.steps.read_text().splitlines() if line.strip()]
    result = summarize_batches(rows, args.coverage)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()

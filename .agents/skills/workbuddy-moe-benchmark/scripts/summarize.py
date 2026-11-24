# SPDX-License-Identifier: Apache-2.0
"""Summarize normalized, already audited rounds. Python standard library only."""
import argparse
import itertools
import json
import math
import statistics
from pathlib import Path


def distribution(values):
    """Descriptive statistics, not independent-sample inference."""
    if not values:
        return {'n': 0}
    median = statistics.median(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {'n': len(values), 'median': median, 'mean': mean,
            'sample_sd': sd, 'min': min(values), 'max': max(values),
            'range_over_median': (max(values) - min(values)) / median,
            'cv': sd / mean if sd is not None else None}


def summarize(plan, rows):
    fingerprints = {r['config_id'] for r in rows if r.get('config_id')}
    if len(fingerprints) > 1:
        raise ValueError('Mixed configuration fingerprints; summarize each configuration separately')
    seeds = plan['seeds']
    policies = plan['policies']
    concurrencies = plan['concurrencies']
    sessions = plan['sessions_per_seed']
    rounds = plan['rounds_per_session']
    if plan['stage'] not in ('prefill', 'decode'):
        raise ValueError('stage must be prefill or decode')
    for xs in (seeds, policies, concurrencies):
        if not xs or len(set(xs)) != len(xs):
            raise ValueError('Empty or duplicate plan axis')
    if sessions < 1 or rounds < 1 or any(c < 1 for c in concurrencies):
        raise ValueError('Counts must be positive')
    expected = set(itertools.product(policies, concurrencies, seeds,
                                     range(1, sessions+1), range(1, rounds+1)))
    values, rejected, seen, records = {}, {}, set(), []
    for original in rows:
        row = dict(original)
        key = tuple(row[k] for k in ('policy', 'concurrency', 'seed', 'session', 'round'))
        phase = row.setdefault('phase', 'formal')
        attempt = row.setdefault('attempt', 1)
        if phase not in ('formal', 'warmup'):
            raise ValueError('phase must be formal or warmup')
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise ValueError('attempt must be a positive integer')
        identity = (phase, attempt, *key)
        if identity in seen:
            raise ValueError(f'Duplicate measurement: {identity}')
        seen.add(identity)
        axis_ok = any(k[:4] == key[:4] for k in expected)
        if row['stage'] != plan['stage'] or not axis_ok or (phase == 'formal' and key not in expected):
            raise ValueError(f'Unexpected measurement: {key}')
        value = row.get('ktokens_s')
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                 or not math.isfinite(value) or value <= 0):
            raise ValueError(f'Invalid throughput: {identity}')
        included = row.setdefault('included', phase == 'formal' and row['status'] == 'ok')
        if not isinstance(included, bool):
            raise TypeError('included must be boolean')
        if included and (phase != 'formal' or row['status'] != 'ok' or value is None):
            raise ValueError('Only completed formal measurements with throughput may be included')
        if phase == 'formal' and not included:
            if not row.get('exclusion_reason'):
                if row['status'] == 'ok':
                    raise ValueError('Exclusion requires an evidence-based reason')
                row['exclusion_reason'] = row['status']
            rejected[key] = row['exclusion_reason']
        if included:
            if key in values:
                raise ValueError(f'Multiple included attempts for {key}')
            values[key] = value
        records.append(row)
    results = []
    for policy, conc in itertools.product(policies, concurrencies):
        keys = {k for k in expected if k[:2] == (policy, conc)}
        missing = sorted(keys - values.keys())
        result = {'policy': policy, 'concurrency': conc,
                      'status': 'incomplete' if missing else 'complete',
                      'valid_rounds': len(keys & values.keys()), 'expected_rounds': len(keys),
                      'missing_or_invalid': [{'seed': k[2], 'session': k[3], 'round': k[4],
                                               'reason': rejected.get(k, 'missing')} for k in missing]}
        result['session_statistics'] = [
            {'seed': seed, 'session': sess,
             'status': 'complete' if len(v) == rounds else 'incomplete',
             'rounds': [{'round': rnd, 'ktokens_s': values[(policy, conc, seed, sess, rnd)]}
                        for rnd in range(1, rounds + 1)
                        if (policy, conc, seed, sess, rnd) in values],
             **distribution(v)}
            for seed in seeds for sess in range(1, sessions + 1)
            for v in [[values[(policy, conc, seed, sess, rnd)]
                       for rnd in range(1, rounds + 1)
                       if (policy, conc, seed, sess, rnd) in values]]]
        if not missing:
            by_seed = {}
            for seed in seeds:
                by_seed[seed] = [statistics.median(values[(policy, conc, seed, sess, rnd)]
                                  for rnd in range(1, rounds+1)) for sess in range(1, sessions+1)]
            seed_medians = {seed: statistics.median(v) for seed, v in by_seed.items()}
            samples = list(seed_medians.values()) if len(seeds) > 1 else by_seed[seeds[0]]
            result.update(median_ktokens_s=statistics.median(samples),
                          stddev_ktokens_s=statistics.stdev(samples) if len(samples) > 1 else None,
                          ddof=1, sample_count=len(samples),
                          statistic_level='seed medians' if len(seeds) > 1 else 'session medians',
                          seed_medians=seed_medians, session_medians=by_seed,
                          between_session_by_seed={seed: distribution(v) for seed, v in by_seed.items()})
        results.append(result)
    return {'status': 'complete' if all(r['status']=='complete' for r in results) else 'incomplete',
                'stage': plan['stage'], 'units': 'K input tokens/s' if plan['stage']=='prefill' else 'K output tokens/s',
                'raw_audit_performed': False, 'round_records': records, 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--rounds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    rows = [json.loads(line) for line in args.rounds.read_text().splitlines() if line.strip()]
    result = summarize(plan, rows)
    # Do not overwrite an earlier report or accidentally overwrite the input.
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'conditions': len(result['results'])}))


if __name__ == '__main__':
    main()

# SPDX-License-Identifier: Apache-2.0
"""Python 3.11+: pinned WorkBuddy data and deterministic instruction workloads.

Download requires curl; prepare requires transformers and a local tokenizer.
Archives and task instructions are read as data, never executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request

OUT = None
TOKENIZER = None
REVISION = '3e8ad839685c589d83a1dd60fe2d15eabb75f27a'
GIT_REVISION = '625b2233093ae4f23e76be28c1f341d41cc70373'
SUBSETS = {'code': (80, 16, 308), 'web': (70, 14, 269),
           'office': (50, 10, 192), 'sec': (60, 12, 231)}
SEEDS = (8, 28, 2026)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def download():
    directory = OUT / 'source/archives'
    directory.mkdir(parents=True, exist_ok=True)
    base = f'https://huggingface.co/datasets/tencent/workbuddy-bench/resolve/{REVISION}'
    sums = directory / 'SHA256SUMS'
    subprocess.run(['curl', '--http1.1', '-fL', '--connect-timeout', '30', '--max-time', '1800',
                    '--retry', '5', base + '/SHA256SUMS', '-o', str(sums)], check=True)
    checksums = {line.split()[1].lstrip('*'): line.split()[0]
                 for line in sums.read_text().splitlines() if line.strip()}
    entries = []
    for subset in SUBSETS:
        name = f'wb-bench-{subset}-v1.0.tar.gz'
        path = directory / name
        expected = checksums[name]
        if not path.exists() or sha(path) != expected:
            part = path.with_suffix('.part')
            subprocess.run(['curl', '--http1.1', '-fL', '--connect-timeout', '30', '--max-time', '1800',
                            '--retry', '5', base + '/' + name,
                            '-o', str(part)], check=True)
            assert sha(part) == expected, name
            part.replace(path)
        target = OUT / 'source/datasets' / name.removesuffix('.tar.gz')
        marker = target / '.archive-sha256'
        if not marker.exists():
            # Extract data only; never execute task scripts or unpack workspace archives.
            stage = OUT / 'source/extract-stage'
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir()
            with tarfile.open(path) as archive:
                for member in archive.getmembers():
                    dest = (stage / member.name).resolve()
                    assert dest.is_relative_to(stage.resolve()), member.name
                    assert member.isfile() or member.isdir(), member.name
                archive.extractall(stage)
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                shutil.rmtree(target)
            (stage / target.name).replace(target)
            marker.write_text(expected + '\n')
            shutil.rmtree(stage)
        assert marker.read_text().strip() == expected
        entries.append(dict(subset=subset, archive=str(path), sha256=expected, path=str(target)))
    write(OUT / 'source.json', dict(dataset_revision=REVISION, reference_repository_revision=GIT_REVISION,
                                   checksum_file_sha256=sha(sums), archives=entries))

def split_tasks(rows):
    calibration, test = {}, {}
    for subset, (count, reserved, _) in SUBSETS.items():
        group = [r for r in rows if r['subset'] == subset]
        assert len(group) == count, (subset, len(group), count)
        group.sort(key=lambda r: hashlib.sha256(
            f'workbuddy-prefill-split-v1/{subset}/{r["task_id"]}'.encode()).hexdigest())
        calibration[subset] = sorted(group[:reserved], key=lambda r: r['task_id'])
        test[subset] = sorted(group[reserved:], key=lambda r: r['task_id'])
    assert {r['prompt_sha256'] for g in calibration.values() for r in g}.isdisjoint(
        {r['prompt_sha256'] for g in test.values() for r in g}), 'Duplicate text across splits'
    return calibration, test

def sample(pool, seed):
    rng = random.Random(seed)
    result = []
    for subset, (_, _, count) in SUBSETS.items():
        result.extend(rng.choice(pool[subset]) for _ in range(count))
    rng.shuffle(result)
    return [dict(r, index=i) for i, r in enumerate(result)]

def prepare():
    import tomllib
    from transformers import AutoTokenizer
    get_tokenizer = AutoTokenizer.from_pretrained
    tokenizer = get_tokenizer(str(TOKENIZER), trust_remote_code=True, local_files_only=True)
    source = json.loads((OUT / 'source.json').read_text())
    rows = []
    for entry in source['archives']:
        for task in sorted((Path(entry['path']) / 'tasks').iterdir()):
            if not task.is_dir():
                continue
            path = task / 'instruction.md'
            step = None
            config_path = task / 'task.toml'
            if not path.exists():
                config = tomllib.loads(config_path.read_text())
                step = config['steps'][0]['name']
                path = task / 'steps' / step / 'instruction.md'
                assert path.resolve().is_relative_to(task.resolve())
            # Decode bytes directly to preserve original newlines.
            prompt = path.read_bytes().decode('utf-8')
            assert prompt.strip(), path
            rows.append(dict(subset=entry['subset'], task_id=task.name, initial_step=step,
                             task_config_sha256=sha(config_path),
                             prompt=prompt, prompt_sha256=sha(path), source=str(path),
                             expected_prompt_tokens=len(tokenizer.encode(prompt, add_special_tokens=True))))
    assert len({(r['subset'], r['task_id']) for r in rows}) == 260
    assert all(0 < r['expected_prompt_tokens'] < 16384 for r in rows), 'Context limit exceeded; no truncation allowed'
    calibration, test = split_tasks(rows)
    pools = {'calibration': sample(calibration, 0)}
    pools.update({f'seed{seed}': sample(test, seed) for seed in SEEDS})
    workloads = {}
    for name, records in pools.items():
        path = OUT / 'workloads' / f'{name}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
        if path.exists():
            assert path.read_text() == text, f'Existing workload changed: {path}'
        else:
            path.write_text(text)
        sizes = sorted(r['expected_prompt_tokens'] for r in records)
        workloads[name] = dict(path=str(path), sha256=sha(path), count=len(records),
            total_input_tokens=sum(sizes), min_input_tokens=sizes[0], max_input_tokens=sizes[-1],
            median_input_tokens=(sizes[499] + sizes[500]) / 2,
            unique_tasks=len({(r['subset'], r['task_id']) for r in records}))
    write(OUT / 'tasks.json', rows)
    write(OUT / 'assets.json', dict(status='ok', source=source, seeds=SEEDS,
        python=sys.version, calibration_seed=0, subsets=SUBSETS,
        instruction_selection='Top-level instruction.md, or first declared step for multi-step tasks. No concatenation or later-step context.',
        tokenizer_files={n: sha(TOKENIZER / n) for n in ('tokenizer.json', 'tokenizer_config.json')},
        split={name: {k: [r['task_id'] for r in v] for k, v in pool.items()}
               for name, pool in [('calibration', calibration), ('test', test)]},
        workloads=workloads, instruction_tokens=dict(min=min(r['expected_prompt_tokens'] for r in rows),
                                                    max=max(r['expected_prompt_tokens'] for r in rows))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['download', 'prepare'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path)
    args = parser.parse_args()
    OUT = args.out.resolve()
    TOKENIZER = args.tokenizer.resolve() if args.tokenizer else None
    if args.phase == 'prepare' and TOKENIZER is None:
        parser.error('prepare requires --tokenizer pointing to local model files')
    download() if args.phase == 'download' else prepare()

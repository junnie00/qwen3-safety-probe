#!/usr/bin/env python3
"""在可验证的用户级 cgroup 中执行本 PoC；拒绝无硬限额运行。"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
CGROOT = Path('/sys/fs/cgroup')


def cgroup_path():
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            return CGROOT / line[3:].lstrip('/')
    raise RuntimeError('需要 cgroup v2')


def memory_info():
    return {line.split(':')[0]: int(line.split()[1]) * 1024
            for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}


def effective_limits():
    cpu = float(len(os.sched_getaffinity(0)))
    mem = memory_info()['MemTotal']
    p = cgroup_path()
    while p != CGROOT.parent:
        if (p / 'cpu.max').exists():
            quota, period = (p / 'cpu.max').read_text().split()
            if quota != 'max':
                cpu = min(cpu, int(quota) / int(period))
        if (p / 'memory.max').exists():
            value = (p / 'memory.max').read_text().strip()
            if value != 'max':
                mem = min(mem, int(value))
        if p == CGROOT:
            break
        p = p.parent
    return cpu, mem


def read_stat(path):
    if not path.exists():
        return None
    value = path.read_text().strip()
    try:
        return int(value)
    except ValueError:
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--cpu', type=float)
    parser.add_argument('--memory', type=int)
    parser.add_argument('--cpus')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('缺少命令')
    log = Path(args.log_dir).resolve()
    log.mkdir(parents=True, exist_ok=True)
    if not args.inside:
        # 所有本包装器运行共享一把锁，防止多个scope各自领取一份限额。
        lock = open(ROOT / 'artifacts' / '.modernbert-poc.lock', 'w')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('已有本 PoC 限额任务运行，拒绝并行领取资源。')
        cpu_total, mem_total = effective_limits()
        cpu = math.floor(min(2.0, cpu_total * 0.30) * 100) / 100
        mem = min(6 * 1024**3, int(mem_total * 0.30) // 4096 * 4096)
        if cpu <= 0 or mem < 1024**3:
            raise SystemExit('可用限额不足，停止。')
        if memory_info()['MemAvailable'] < mem + 1024**3:
            raise SystemExit('当前可用内存不足以保留系统余量，停止。')
        cpus = ','.join(map(str, sorted(os.sched_getaffinity(0))[:max(1, math.ceil(cpu))]))
        unit = 'modernbert-poc-' + uuid.uuid4().hex[:10]
        cmd = ['systemd-run', '--user', '--scope', '--quiet', '--unit=' + unit,
               '-p', f'CPUQuota={cpu * 100:g}%', '-p', f'MemoryMax={mem}',
               '-p', f'MemoryHigh={mem * 5 // 6}', '-p', 'MemorySwapMax=0',
               '-p', 'MemoryAccounting=yes', '-p', 'CPUAccounting=yes',
               'taskset', '-c', cpus, sys.executable, str(Path(__file__).resolve()),
               '--inside', '--cpu', str(cpu), '--memory', str(mem), '--cpus', cpus,
               '--log-dir', str(log), '--', *command]
        result = subprocess.run(cmd, cwd=ROOT)
        return result.returncode

    p = cgroup_path()
    quota, period = (p / 'cpu.max').read_text().split()
    actual_mem = read_stat(p / 'memory.max')
    actual_high = read_stat(p / 'memory.high')
    affinity = sorted(os.sched_getaffinity(0))
    if (quota == 'max' or int(quota) / int(period) > args.cpu + 1e-6
            or not isinstance(actual_mem, int) or actual_mem > args.memory
            or not isinstance(actual_high, int) or actual_high > args.memory * 5 // 6
            or read_stat(p / 'memory.swap.max') != 0
            or affinity != [int(x) for x in args.cpus.split(',')]):
        raise SystemExit('cgroup/affinity 硬限核验失败，拒绝执行负载。')
    proof = {'cgroup': str(p), 'cpu.max': f'{quota} {period}',
             'memory.max': actual_mem, 'memory.high': actual_high,
             'memory.swap.max': 0, 'affinity': affinity, 'command': command,
             'verified_at': time.time()}
    (log / 'resource_limits.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    print('已验证硬限额: ' + json.dumps(proof, ensure_ascii=False), flush=True)
    env = os.environ.copy()
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        env[key] = str(max(1, min(2, len(affinity))))
    env.update(TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_PROGRESS_BARS='1',
               HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_XET='1',
               HF_HUB_DOWNLOAD_TIMEOUT='120', PYTHONUNBUFFERED='1',
               POC_CGROUP=str(p))
    child = subprocess.Popen(['nice', '-n', '15', 'ionice', '-c', '3', *command],
                             cwd=ROOT, env=env)
    with (log / 'resources.jsonl').open('a') as f:
        while True:
            snap = {'time': time.time()}
            for name in ['memory.current', 'memory.peak', 'memory.events', 'cpu.stat']:
                snap[name] = read_stat(p / name)
            f.write(json.dumps(snap) + '\n')
            f.flush()
            status = child.poll()
            if status is not None:
                break
            time.sleep(2)
    (log / 'exit.json').write_text(json.dumps({'exit_code': status, 'time': time.time()}))
    return status


if __name__ == '__main__':
    sys.exit(main())

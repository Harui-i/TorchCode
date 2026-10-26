"""Local CLI judge. Test definitions are data; upstream modules are never imported."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / 'data/progress.json'


def load_progress():
    return json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}


def save_result(task_id, solved, elapsed):
    progress = load_progress()
    entry = progress.setdefault(task_id, {})
    entry['attempts'] = entry.get('attempts', 0) + 1
    if solved:
        entry['status'] = 'solved'
        entry['solved_at'] = datetime.now().isoformat()
        previous = entry.get('best_time')
        entry['best_time'] = min(previous, elapsed) if previous is not None else elapsed
    elif entry.get('status') != 'solved':
        entry['status'] = 'attempted'
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROGRESS.with_suffix('.tmp')
    temporary.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(PROGRESS)


def grade(task_id, task, path, record):
    print(f"{task_id}: {task['title']} ({path})", flush=True)
    start = time.perf_counter()
    try:
        namespace = runpy.run_path(str(path), run_name='__answer__')
        function = namespace[task['function_name']]
    except Exception:
        traceback.print_exc()
        if record:
            save_result(task_id, False, 0)
        return 1
    passed = 0
    for test in task['tests']:
        code = test['code'].replace('{fn}', task['function_name'])
        try:
            exec(compile(code, f"<test:{test['name']}>", 'exec'),
                 {task['function_name']: function})
        except Exception as error:
            print(f"  FAIL {test['name']}: {type(error).__name__}: {error}", flush=True)
        else:
            passed += 1
            print(f"  PASS {test['name']}", flush=True)
    solved = passed == len(task['tests'])
    elapsed = time.perf_counter() - start
    print(f"{passed}/{len(task['tests'])} passed ({elapsed:.2f}s)", flush=True)
    if record:
        save_result(task_id, solved, elapsed)
    return 0 if solved else 1


def main():
    parser = argparse.ArgumentParser(description='PyTorch 学習・採点 CLI')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='問題一覧')
    commands.add_parser('status', help='進捗一覧')
    hint = commands.add_parser('hint', help='ヒントを表示')
    hint.add_argument('task')
    check = commands.add_parser('check', help='解答 .py を採点')
    check.add_argument('task', help='例: relu または 01_relu')
    check.add_argument('--file', type=Path, help='別の解答ファイルを採点')
    check.add_argument('--timeout', type=float, default=60, help='全体の制限秒数 (既定: 60)')
    check.add_argument('--no-progress', action='store_true', help='進捗を更新しない')
    check.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    tasks = json.loads((ROOT / 'judge/tasks.json').read_text())
    if args.command in ('list', 'status'):
        progress = load_progress() if args.command == 'status' else {}
        for task_id, task in sorted(tasks.items(), key=lambda item: item[1]['exercise']):
            state = progress.get(task_id, {}).get('status', 'todo')
            print(f"{state:9} {task_id:24} {task['difficulty']:6} exercises/{task['exercise']}")
        return 0
    aliases = {Path(task['exercise']).stem: key for key, task in tasks.items()}
    task_id = aliases.get(args.task, args.task)
    if task_id not in tasks:
        parser.error(f'不明な問題: {args.task} (list で確認できます)')
    task = tasks[task_id]
    if args.command == 'hint':
        print(task['hint'])
        return 0
    if args.timeout <= 0:
        parser.error('--timeout は正の秒数を指定してください')
    path = (args.file or ROOT / 'exercises' / task['exercise']).resolve()
    if not path.is_file():
        parser.error(f'解答ファイルがありません: {path}')
    if args.worker:
        return grade(task_id, task, path, not args.no_progress)
    # Separate process allows a stuck solution/test to be terminated. This is not a sandbox.
    command = [sys.executable, str(Path(__file__).resolve()), 'check', task_id,
               '--file', str(path), '--worker']
    if args.no_progress:
        command.append('--no-progress')
    env = dict(os.environ)
    env.setdefault('OMP_NUM_THREADS', '2')
    env.setdefault('MKL_NUM_THREADS', '2')
    try:
        return subprocess.run(command, timeout=args.timeout, env=env).returncode
    except subprocess.TimeoutExpired:
        print(f'採点を中断しました: {args.timeout:g} 秒を超過', file=sys.stderr)
        if not args.no_progress:
            save_result(task_id, False, 0)
        return 124


if __name__ == '__main__':
    sys.exit(main())

"""Bound one Linux command and its private descendants, including new sessions.

A dedicated supervisor becomes a child subreaper before launching the command.
It has no unrelated children, so adopted Ray daemons remain unambiguously owned.
Kernel-uninterruptible processes can survive SIGKILL; report them explicitly.
A scheduler/container cgroup remains the strongest external deadline boundary.
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RunResult:
    returncode: int
    timed_out: bool
    elapsed: float
    cleanup_errors: list[str]


def _processes(root, tracked):
    import psutil
    try:
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in descendants:
        try:
            tracked[(child.pid, child.create_time())] = child
        except psutil.NoSuchProcess:
            pass
    live = []
    for process in tracked.values():
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                live.append(process)
        except psutil.NoSuchProcess:
            pass
    return live


def _signal(processes, signum, errors):
    import psutil
    for process in reversed(processes):
        try:
            # psutil checks PID reuse against the retained creation time.
            process.send_signal(signum)
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as exc:
            errors.append(f'pid={process.pid}: {type(exc).__name__}')


def _cleanup(root, tracked, deadline, errors):
    term_until = time.monotonic() + max(0, deadline - time.monotonic()) / 2
    sent = set()
    while time.monotonic() < deadline:
        live = _processes(root, tracked)
        if not live:
            return
        signum = signal.SIGTERM if time.monotonic() < term_until else signal.SIGKILL
        pending = []
        for process in live:
            try:
                if (process.pid, process.create_time(), signum) not in sent:
                    pending.append(process)
            except Exception:
                continue
        _signal(pending, signum, errors)
        for process in pending:
            try:
                sent.add((process.pid, process.create_time(), signum))
            except Exception:
                pass
        time.sleep(min(.05, max(0, deadline - time.monotonic())))
    live = _processes(root, tracked)
    _signal(live, signal.SIGKILL, errors)
    if live:
        errors.append('Processes still live at cleanup deadline: ' + ','.join(str(p.pid) for p in live))


def _supervise(request):
    import psutil
    started = request['started']
    deadline = started + request['timeout_s']
    reserve = min(request['cleanup_s'], request['timeout_s'] / 2)
    execution_deadline = deadline - reserve
    errors, tracked = [], {}
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child = None
    code, expired = 1, False
    supervisor = psutil.Process()
    try:
        if sys.platform != 'linux':
            raise RuntimeError('Hard-deadline process supervision requires Linux')
        # PR_SET_CHILD_SUBREAPER: orphaned descendants return to this private
        # supervisor even when they called setsid or double-forked.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
        with open(request['log_path'], 'ab', buffering=0) as log:
            child = subprocess.Popen(request['command'], cwd=request['cwd'], env=request['env'],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            tracked[(child.pid, psutil.Process(child.pid).create_time())] = psutil.Process(child.pid)
            while True:
                _processes(supervisor, tracked)
                status = child.poll()
                if status is not None:
                    code = status
                    break
                if stopped or time.monotonic() >= execution_deadline:
                    expired = not stopped
                    code = 124 if expired else 130
                    break
                time.sleep(min(.05, max(0, execution_deadline - time.monotonic())))
    except BaseException as exc:
        errors.append(f'Supervisor: {type(exc).__name__}: {exc}')
    finally:
        _cleanup(supervisor, tracked, min(deadline, time.monotonic() + reserve), errors)
        if child is not None:
            try:
                child.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass
        # Reap adopted zombies after the native process has been polled.
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if not pid:
                    break
            except ChildProcessError:
                break
    if errors and code == 0:
        code = 1
    return RunResult(code, expired, time.monotonic() - started, errors)


def run(command, cwd, env, log_path, timeout_s, cleanup_s=15):
    """Run a command within timeout_s total seconds, including cleanup reserve.

On KeyboardInterrupt, clean the private process tree and re-raise. A JSON sidecar
next to log_path preserves the cleanup outcome. No global Ray/pkill commands run.
"""
    if timeout_s <= 0 or cleanup_s <= 0:
        raise ValueError('timeout_s and cleanup_s must be positive')
    if not command:
        raise ValueError('command must not be empty')
    started = time.monotonic()
    request = dict(command=list(command), cwd=str(cwd) if cwd is not None else None,
                   env=dict(env) if env is not None else dict(os.environ),
                   log_path=str(Path(log_path).resolve()), timeout_s=float(timeout_s),
                   cleanup_s=float(cleanup_s), started=started)
    Path(request['log_path']).parent.mkdir(parents=True, exist_ok=True)
    worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--supervise'],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              start_new_session=True, text=True)
    interrupted = False
    tracked = {}
    import psutil
    root = psutil.Process(worker.pid)
    payload = json.dumps(request)
    limit = started + timeout_s + 2
    output = diagnostic = ''
    try:
        while True:
            _processes(root, tracked)
            try:
                output, diagnostic = worker.communicate(input=payload, timeout=.1)
                break
            except subprocess.TimeoutExpired:
                payload = None
            except KeyboardInterrupt:
                interrupted = True
                payload = None
                worker.send_signal(signal.SIGTERM)
                limit = min(limit, time.monotonic() + cleanup_s + 1)
            if time.monotonic() >= limit:
                raise subprocess.TimeoutExpired(command, timeout_s)
        try:
            result = RunResult(**json.loads(output))
        except (ValueError, TypeError):
            result = RunResult(1, False, time.monotonic() - started,
                               ['Supervisor did not return a valid outcome: ' + diagnostic[-1000:]])
    except BaseException as exc:
        errors = [f'Outer supervisor watchdog: {type(exc).__name__}']
        # Snapshot before signaling the supervisor, preserving ownership of
        # descendants if the supervisor exits before they do.
        _processes(root, tracked)
        tracked[(root.pid, root.create_time())] = root
        _cleanup(root, tracked, time.monotonic() + min(2, cleanup_s), errors)
        result = RunResult(124, True, time.monotonic() - started, errors)
        if isinstance(exc, KeyboardInterrupt):
            interrupted = True
    finally:
        for stream in (worker.stdin, worker.stdout, worker.stderr):
            if stream:
                stream.close()
    sidecar = Path(str(log_path) + '.deadline.json')
    sidecar.write_text(json.dumps(asdict(result), indent=2) + '\n')
    if interrupted:
        raise KeyboardInterrupt
    return result


if __name__ == '__main__':
    if sys.argv[1:] != ['--supervise']:
        raise SystemExit('This module is called through benchkit.deadline.run')
    result = _supervise(json.load(sys.stdin))
    print(json.dumps(asdict(result)), flush=True)

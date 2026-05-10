from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

Command = Union[Sequence[str], str]


def _command_to_text(cmd: Command) -> str:
    if isinstance(cmd, str):
        return cmd
    return ' '.join(shlex.quote(str(x)) for x in cmd)


def run_streaming(
    cmd: Command,
    log_path: Optional[Union[str, Path]] = None,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Mapping[str, str]] = None,
    check: bool = True,
    heartbeat_seconds: float = 60.0,
) -> int:
    """Run a command and mirror every output line to both notebook and log file.

    Colab sometimes sends child-process output only to the runtime log when the
    command is launched through shell piping. This function avoids that by using
    subprocess.Popen, reading stdout line by line in Python, printing each line
    with flush=True, and writing the exact same line into the Drive log file.
    """
    run_env = os.environ.copy()
    run_env['PYTHONUNBUFFERED'] = '1'
    run_env['PYTHONIOENCODING'] = 'utf-8'
    if env:
        run_env.update({str(k): str(v) for k, v in env.items()})

    cmd_text = _command_to_text(cmd)
    print('=' * 80, flush=True)
    print('[run_streaming] command:', cmd_text, flush=True)
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print('[run_streaming] log file:', log_path, flush=True)
    print('=' * 80, flush=True)

    log_fh = None
    try:
        if log_path is not None:
            log_fh = open(log_path, 'a', encoding='utf-8', buffering=1)
            log_fh.write('=' * 80 + '\n')
            log_fh.write(f'[run_streaming] command: {cmd_text}\n')
            log_fh.write('=' * 80 + '\n')

        if isinstance(cmd, str):
            proc = subprocess.Popen(
                ['bash', '-lc', cmd],
                cwd=str(cwd) if cwd is not None else None,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        else:
            proc = subprocess.Popen(
                [str(x) for x in cmd],
                cwd=str(cwd) if cwd is not None else None,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

        assert proc.stdout is not None
        q: queue.Queue[str | None] = queue.Queue()

        def _reader() -> None:
            try:
                for item in proc.stdout:
                    q.put(item)
            finally:
                q.put(None)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        last_output = time.monotonic()
        reader_done = False
        while not reader_done:
            try:
                line = q.get(timeout=min(max(float(heartbeat_seconds), 1.0), 60.0))
            except queue.Empty:
                if proc.poll() is None and time.monotonic() - last_output >= float(heartbeat_seconds):
                    msg = '[run_streaming] still running; no child output yet. This usually means the first dataloader/model step is still working.\n'
                    print(msg, end='', flush=True)
                    if log_fh is not None:
                        log_fh.write(msg)
                        log_fh.flush()
                    last_output = time.monotonic()
                continue
            if line is None:
                reader_done = True
                break
            last_output = time.monotonic()
            print(line, end='', flush=True)
            if log_fh is not None:
                log_fh.write(line)
                log_fh.flush()
        proc.wait()
        rc = int(proc.returncode)
    finally:
        if log_fh is not None:
            log_fh.close()

    print(f'[run_streaming] return_code={rc}', flush=True)
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc

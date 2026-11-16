"""Wall-clock containment for the graded workflow only (no optimization code).

Windows process.wait(timeout) excludes standby time. Poll an absolute deadline
instead; keep scientific work and filesystem telemetry off the monitoring loop.
All children enter a kill-on-close Job before their start gate is released.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent


class Deadline:
    """Count suspended time; a backwards civil-clock adjustment cannot extend it."""
    def __init__(self, seconds):
        if not isinstance(seconds,(int,float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("A finite positive wall-clock budget is required")
        self.seconds = float(seconds)
        self.started_at = time.time()
        self.monotonic_start = time.monotonic()

    @property
    def elapsed(self):
        return max(0., time.time()-self.started_at, time.monotonic()-self.monotonic_start)

    @property
    def remaining(self):
        return max(0., self.seconds-self.elapsed)


@contextmanager
def prevent_idle_sleep(enabled=True):
    """Temporary request only: display/lid policy and permanent power plan unchanged."""
    state = dict(requested=enabled, active=False)
    previous = None
    if enabled and os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
        kernel.SetThreadExecutionState.restype = ctypes.c_uint
        previous = kernel.SetThreadExecutionState(0x80000001)  # CONTINUOUS | SYSTEM_REQUIRED
        state['active'] = bool(previous)
        if not previous:
            state['error'] = str(ctypes.WinError(ctypes.get_last_error()))
    try:
        yield state
    finally:
        if previous:
            kernel.SetThreadExecutionState(previous)


def _telemetry(messages, stop, directory, phase):
    """Best effort daemon: slow OneDrive/log reads never block deadline enforcement."""
    cache = {}
    log_path = Path(directory)/f'heartbeat_{phase}_graded_max_reliability.jsonl'
    while True:
        try:
            item = messages.get(timeout=.2)
        except queue.Empty:
            if stop.is_set(): return
            continue
        try:
            # Native evaluation records contain the worker PID and timestamps.
            for path in (Path(directory)/'evaluations').glob('*_hpp_sizing_base.json'):
                stamp = path.stat().st_mtime_ns
                if path not in cache or cache[path][0] != stamp:
                    try:
                        record = json.loads(path.read_text(encoding='utf8'))
                        cache[path] = stamp, record
                    except (ValueError, OSError):
                        continue
            completed = [r for _, r in cache.values() if 'completed_at' in r]
            last = max(completed, key=lambda r:r['completed_at'], default={})
            item.update(completed_evaluations=len(completed),
                        observed_worker_pids=sorted({r['pid'] for _,r in cache.values() if r.get('pid')}),
                        last_completed_candidate={k:last.get(k) for k in
                            ('completed_at','status','design','lcoe_hpp_eur_per_mwh')})
            native_log = Path(directory)/'native_ego_hpp_sizing_base.log'
            if native_log.exists():
                with native_log.open('rb') as stream:
                    stream.seek(max(0, native_log.stat().st_size-65536))
                    iterations = re.findall(r'Iteration (\d+) took', stream.read().decode('utf8',errors='replace'))
                item['last_completed_ego_iteration'] = int(iterations[-1]) if iterations else None
            with log_path.open('a', encoding='utf8') as stream:
                stream.write(json.dumps(item)+'\n')
        except Exception:
            # Diagnostic I/O cannot disable timeout enforcement.
            pass


def run_contained(command, run_dir, limit_seconds, gate_path, *, cleanup_seconds=10.,
                  heartbeat_seconds=10., prevent_sleep=True, phase='search'):
    """Bound the whole process tree; never wait on EGO's Pool, queues or results.

    No Python controller can execute while the OS is asleep or unresponsive.
    After resume, an expired deadline is enforced on the next <=0.1 s poll.
    OS termination failures are reported explicitly instead of waiting forever.
    """
    budget = Deadline(limit_seconds)
    Deadline(cleanup_seconds)  # Validate before launching anything.
    if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
        raise ValueError('heartbeat_seconds must be positive and finite')
    job = process = None
    messages = queue.Queue(maxsize=4)
    stop_telemetry = threading.Event()
    telemetry = threading.Thread(target=_telemetry,args=(messages,stop_telemetry,run_dir,phase),daemon=True)
    reason, error, cleanup_error = 'NORMAL_EXIT', None, None
    cleanup = {}
    def emit(event, **extra):
        item = dict(event=event, phase=phase, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    controller_pid=os.getpid(), outer_pid=process.pid if process else None,
                    started_at=budget.started_at, deadline_epoch=budget.started_at+budget.seconds,
                    elapsed_wall_seconds=budget.elapsed, remaining_seconds=budget.remaining, **extra)
        try:
            messages.put_nowait(item)
        except queue.Full:
            pass
    env = os.environ.copy()
    env.update(OPENMDAO_REPORTS='0', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    env.setdefault('OMP_NUM_THREADS','1')
    name = 'native_ego_hpp_sizing_base.log' if phase == 'search' else f'{phase}_graded_max_reliability.log'
    # Opening the log precedes the child: no uncontrolled optimizer exists yet.
    with prevent_idle_sleep(prevent_sleep) as sleep_state, (Path(run_dir)/name).open('w',encoding='utf8') as log:
        telemetry.start()
        try:
            if os.name == 'nt':
                import win32api, win32con, win32job
                job = win32job.CreateJobObject(None, '')
                info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
                info['BasicLimitInformation']['LimitFlags'] = win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,
                start_new_session=os.name!='nt')
            if job is not None:
                handle = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA|win32con.PROCESS_TERMINATE,False,process.pid)
                try:
                    win32job.AssignProcessToJobObject(job,handle)
                finally:
                    handle.Close()
            Path(gate_path).touch()
            emit('START', idle_sleep_prevention=sleep_state)
            heartbeat = 0.
            while True:
                # Check deadline before return status: expiry is never extended.
                if budget.remaining <= 0:
                    reason = 'TIME_LIMIT'
                    break
                if process.poll() is not None:
                    break
                if budget.elapsed >= heartbeat:
                    emit('HEARTBEAT')
                    heartbeat = budget.elapsed+heartbeat_seconds
                time.sleep(min(.1,budget.remaining))
        except KeyboardInterrupt:
            reason = 'USER_INTERRUPTED'
        except Exception as exc:
            reason, error = 'PROCESS_ERROR', f'{type(exc).__name__}: {exc}'
        finally:
            elapsed = budget.elapsed
            clean = Deadline(cleanup_seconds)
            try:
                if job is not None:
                    win32job.TerminateJobObject(job,124 if reason=='TIME_LIMIT' else 1)
                    while True:
                        info = win32job.QueryInformationJobObject(job,win32job.JobObjectBasicAccountingInformation)
                        cleanup = dict(mechanism='Windows Job Object',total_processes=info['TotalProcesses'],
                                       active_processes_after_stop=info['ActiveProcesses'])
                        if info['ActiveProcesses']==0 or clean.remaining<=0:
                            break
                        time.sleep(min(.05,clean.remaining))
                elif process is not None and os.name!='nt':
                    try:
                        os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    cleanup = dict(mechanism='POSIX process group')
                elif process is not None and process.poll() is None:
                    process.kill()  # Assignment failed; child still behind gate.
            except Exception as exc:
                cleanup_error = f'{type(exc).__name__}: {exc}'
            finally:
                if job is not None:
                    job.Close()  # Independent kill-on-close fallback, no blocking join.
                if process is not None:
                    if process.poll() is None:
                        try:
                            process.kill()
                        except OSError as exc:
                            cleanup_error = cleanup_error or str(exc)
                    while process.poll() is None and clean.remaining>0:
                        time.sleep(min(.05,clean.remaining))
                cleanup['root_stopped'] = process is None or process.poll() is not None
                cleanup['verified'] = (cleanup['root_stopped'] and
                    cleanup.get('active_processes_after_stop',0)==0 and cleanup_error is None)
                cleanup['termination_method'] = 'forced_tree_cleanup'
                cleanup['error'] = cleanup_error
                clean_elapsed = clean.elapsed
            emit('END', termination_reason=reason, process_cleanup=cleanup, cleanup_seconds=clean_elapsed)
    stop_telemetry.set()
    telemetry.join(timeout=.2)  # Never wait indefinitely for diagnostic disk I/O.
    return dict(termination_reason=reason, optimization_seconds=elapsed, started_at=budget.started_at,
                ended_at=time.time(), deadline_epoch=budget.started_at+budget.seconds,
                cleanup_seconds=clean_elapsed, process_cleanup=cleanup,
                outer_pid=process.pid if process else None, error=error,
                idle_sleep_prevention=sleep_state, returncode=process.returncode if process else None)

"""Publish the staged queue only when the Gemma CTI-Bench workers have exited."""
import fcntl
import os
from pathlib import Path
import time
import traceback
from common import OUT, read, save
from host_lifecycle import sha, ticks

def live(record):
    try:
        fields = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return int(fields[19]) == record['start_ticks'] and fields[0] != 'Z'

def state(status, **details):
    save(OUT / 'release_job.json', {'status': status, 'pid': os.getpid(),
        'start_ticks': ticks(os.getpid()), 'at': time.time(), **details})

def main():
    staged = read(OUT / 'queue_after_gemma.json')
    predecessor = Path(staged['predecessor_job'])
    while True:
        current = read(predecessor)
        if live(current):
            state('waiting_for_gemma_ctibench', method=current.get('method'))
            time.sleep(20)
            continue
        assert current['status'] in {'completed', 'completed_with_blockers'}, current['status']
        child = Path(staged['predecessor_child'])
        assert not child.exists() or not live(read(child)), 'Gemma child still active'
        break
    current = read(OUT / 'job.json')
    assert live(current) and current['status'] == 'waiting_for_validated_290_question_adapters'
    assert sha(OUT / 'queue.json') == staged['continuation']['previous_queue_sha256']
    assert sha(OUT / 'results.json') == staged['continuation']['previous_ledger_sha256']
    save(OUT / 'queue.json', staged)
    state('released', supervisor_pid=current['pid'])

if __name__ == '__main__':
    with (OUT / 'release.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception:
            state('blocked', error=traceback.format_exc())
            raise

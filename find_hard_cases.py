"""
Find UDIS-D test pairs where classical stitching (SIFT + RANSAC) fails or hangs.
"""
import cv2
import os
import sys
from pathlib import Path
from multiprocessing import Process, Queue
from tqdm import tqdm

test_dir = Path('UDIS-D/testing')
input1_dir = test_dir / 'input1'
input2_dir = test_dir / 'input2'

TIMEOUT_SECONDS = 30  # classical stitching should finish in under 30s for any reasonable pair

status_names = {
    0: 'OK',
    1: 'NEED_MORE_IMGS',
    2: 'HOMOGRAPHY_FAIL',
    3: 'CAMERA_PARAMS_FAIL',
    -1: 'TIMEOUT',
}


def stitch_worker(img1_path, img2_path, queue):
    """Run stitching in a subprocess so we can kill it if it hangs."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        queue.put(('error', 'could not load'))
        return
    stitcher = cv2.Stitcher.create(mode=cv2.Stitcher_PANORAMA)
    status, _ = stitcher.stitch([img1, img2])
    queue.put(('done', status))


def run_with_timeout(img1_path, img2_path, timeout):
    """Return status code, or -1 if it timed out."""
    q = Queue()
    p = Process(target=stitch_worker, args=(img1_path, img2_path, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return -1  # timeout
    if not q.empty():
        kind, result = q.get()
        if kind == 'done':
            return result
    return -1


if __name__ == '__main__':
    failures = {1: [], 2: [], 3: [], -1: []}
    successes = []

    files = sorted(os.listdir(input1_dir))
    print(f'Testing {len(files)} pairs (timeout = {TIMEOUT_SECONDS}s per pair)\n')

    pbar = tqdm(files, desc='Stitching', unit='pair', smoothing=0.1)
    for fname in pbar:
        status = run_with_timeout(
            str(input1_dir / fname),
            str(input2_dir / fname),
            TIMEOUT_SECONDS
        )

        if status == 0:
            successes.append(fname)
        else:
            failures.setdefault(status, []).append(fname)
            pbar.write(f'  FAIL  {fname}  ->  {status_names.get(status, f"UNKNOWN_{status}")}')

        pbar.set_postfix({
            'OK': len(successes),
            'NeedMore': len(failures[1]),
            'HomogFail': len(failures[2]),
            'CamFail': len(failures[3]),
            'Timeout': len(failures[-1]),
        })

    pbar.close()

    print('\n=== Summary ===')
    print(f'Classical stitcher succeeded: {len(successes)} / {len(files)}  '
          f'({100.0 * len(successes) / len(files):.1f}%)')
    for code, names in failures.items():
        pct = 100.0 * len(names) / len(files)
        print(f'{status_names[code]}: {len(names)}  ({pct:.1f}%)')
        if names:
            print(f'  First 20: {names[:20]}')

    with open('classical_failures.txt', 'w') as f:
        for code, names in failures.items():
            for n in names:
                f.write(f'{status_names[code]}\t{n}\n')
    print('\nSaved to classical_failures.txt')
    print('Slide candidates: NEED_MORE_IMGS (low-texture) and TIMEOUT (ambiguous/repetitive features)')

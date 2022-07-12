#!/usr/bin/env python

"""Download dataset for the Kaggle Landmarks competition.

Features:
- progress bar
- multiple workers
- skip already existing files
- show inbound traffic size
- show number of failed URLs
- write rows with failed URLs to CSV file
"""


import argparse
import csv
import math
import os
import requests
import sys

from contextlib import suppress
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm


SIZE_PREFIXES = tuple(['B'] + [f'{s}iB' for s in 'KMGTPEZY'])
SIZE_POWERS = tuple(2**(10*i) for i in range(len(SIZE_PREFIXES)))

def humansize(n, fmt='{}{}'):
    """Format the size in bytes t human-readable representation."""
    if n < SIZE_POWERS[0]:
        return fmt.format(n, SIZE_PREFIXES[0])
    base = int(math.log2(n)/10)
    unit = SIZE_PREFIXES[base]
    size = n/SIZE_POWERS[base]
    return fmt.format(size, unit)


def download_row(outdir, row, timeout=0):
    """Download a single row from the dataset.

    Returns a boolean success status, the processed row
    and the size of downloaded file.
    """

    fileid, url, label = row
    file = outdir/label/f'{fileid}.jpg'

    try:
        req = requests.get(url, timeout=timeout)
        req.raise_for_status()
    except requests.exceptions.RequestException:
        return False, row, 0

    file.write_bytes(req.content)
    return True, row, len(req.content)


def download(srcfile, outdir, failfile, nworkers=None):
    """Download the Google Landmarks dataset.

    Multiple worker processes are used for downloading.
    Already existing files are skipped.
    On a failed download, the corresponding row is written to a spearate file.
    """

    # Infer the number of workers if it's not explicitly specified.
    if nworkers is None:
        nworkers = len(os.sched_getaffinity(0))
    print(f'Download workers: {nworkers}')

    # Read everything in memory.
    with (srcfile).open() as srcfd:
        header, *data = list(csv.reader(srcfd))

    # Pre-create label directories.
    outdir.mkdir(exist_ok=True)
    for label in set(row[2] for row in data):
        (outdir/label).mkdir(exist_ok=True)

    # Filter out rows with already existing files.
    existing = set(p.name for p in outdir.glob('*/*.jpg'))
    data = list(filter(lambda row: row[0] not in existing, data))
    print(f'Skip existing files: {len(existing)}')

    # We don't use the Python context managers
    # because the indentation level gets too deep.

    failfd = failfile.open('w')
    pool = Pool(processes=nworkers)
    pbar = tqdm(data, total=len(data), unit='files')

    try:
        # Write CSV header in the failed rows file.
        failcsv = csv.writer(failfd)
        failcsv.writerow(header)
        failfd.flush()

        # Define shortcut functions.
        downloadfunc = partial(download_row, outdir, timeout=2)
        sizefunc = partial(humansize, fmt='{:.2f}{}')

        # Initial values for displayed stats.
        inbound = 0
        failed = 0

        # Process rows using multiple workers from the pool.
        for ok, row, size in pool.imap_unordered(downloadfunc, data):
            if ok:
                inbound += size
            else:
                failcsv.writerow(row)
                failfd.flush()
                failed += 1
            # Update the progress bar.
            pbar.set_postfix(failed=str(failed), inbound=sizefunc(inbound))
            pbar.update()

    finally:
        pbar.close()
        pool.close()
        failfd.close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='Omit WORKERS to use as many as possible.',
    )
    ap.add_argument('srcfile', type=Path,
                    help='source CSV file')
    ap.add_argument('outdir', type=Path,
                    help='output directory')
    ap.add_argument('failfile', type=Path,
                    help='output CSV file for failed rows')
    ap.add_argument('-n', type=int, metavar='WORKERS',
                    help='number of download workers')
    # Print help if no arguments have been supplied.
    if len(sys.argv) == 1:
        ap.print_help(sys.stderr)
        sys.exit(2)
    args = ap.parse_args()
    download(args.srcfile, args.outdir, args.failfile, nworkers=args.n)


if __name__ == '__main__':
    with suppress(KeyboardInterrupt):
        main()

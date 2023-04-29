"""Implement `urllib.urlretrieve(url, filename)` with requests library."""

import contextlib
import os
import urllib

import requests


def urlretrieve(url: str,
                filename: os.PathLike | str) -> tuple[str, dict[str, str]]:
    with contextlib.closing(requests.get(url, stream=True)) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8_192):
                f.write(chunk)

    return filename, r.headers


def urlretrieve(url: str,
                filename: os.PathLike | str) -> tuple[str, dict[str, str]]:
    with contextlib.closing(requests.get(url, stream=True)) as r:
        r.raise_for_status()
        size = int(r.headers.get('Content-Length', '-1'))
        read = 0
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8_192):
                read += len(chunk)
                f.write(chunk)

    if size >= 0 and read < size:
        msg = f'retrieval incomplete: got only {read:d} out of {size:d} bytes'
        raise urllib.ContentTooShortError(msg, (filename, r.headers))

    return filename, r.headers

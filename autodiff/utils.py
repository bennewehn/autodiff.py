# https://stackoverflow.com/a/63831344
def download(url: str, filename: str):
    '''
    Downloads file from url.

    Args:
        url (str): URL
        filename (str): Path to save file.
    '''
    import functools
    import pathlib
    import shutil
    import requests
    from tqdm.auto import tqdm

    print(f"Downloading {url}")
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(
            f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path


def get(url, name, dir):
    import os
    if not os.path.exists(dir):
        os.mkdir(dir)
    path = os.path.join(dir + name)
    if not os.path.isfile(path):
        download(url + name, path)

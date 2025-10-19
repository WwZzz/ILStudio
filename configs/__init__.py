import os
ILSTD_CACHE = os.environ.get('ILSTD_CACHE', os.path.join(os.path.expanduser('~'), ".cache/ilstd"))
os.makedirs(ILSTD_CACHE, exist_ok=True)
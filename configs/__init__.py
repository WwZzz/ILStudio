import os
ILSTD_CACHE = os.environ.get('ILSTD_CACHE', '~/.cache/ilstd')
os.makedirs(ILSTD_CACHE, exist_ok=True)
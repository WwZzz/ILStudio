import os

# Suppress TensorFlow warnings - must be set before TensorFlow is imported anywhere
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # Only show ERROR logs
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # Disable oneDNN messages

ILSTD_CACHE = os.environ.get('ILSTD_CACHE', os.path.join(os.path.expanduser('~'), ".cache/ilstd"))
os.makedirs(ILSTD_CACHE, exist_ok=True)
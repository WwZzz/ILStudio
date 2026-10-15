# Remote inference

This example uses `vla.serve`, `vla.connect_policy`, and `Environment.evaluate`.
The Python files can be edited or copied into your own application. First train
the checkpoint in [Example 01](../01_train_and_eval_act_on_aloha).

## Policy server

Use your existing VLAStudio base environment. From the repository root:

```bash
python examples/02_remote_inference/serve.py
```

Wait for the server to listen on port 5000. Leave this terminal running.

## Simulation client

Open another terminal and activate the same base environment:

```bash
python examples/02_remote_inference/evaluate.py
```

Change the server address inside `evaluate.py` to connect across machines.
The default ACT + ALOHA example needs no extra installation. Both processes
can share base. For a policy or simulator with conflicting dependencies, use
separate prepared environments; the Python commands stay the same.
See [setup](../README.md).

## Supported transports

| Transport | Server address | Evaluation address | Scope |
| --- | --- | --- | --- |
| TCP | `0.0.0.0:5000` | `host:5000` | Default; local or across machines |
| HTTP | `http://0.0.0.0:8000` | `http://host:8000` | JSON/HTTP; across machines |
| HTTPS | `https://0.0.0.0:8443` | `https://host:8443` | Requires certificate configuration |
| SHM | `shm://act_policy` | `shm://act_policy` | Same machine only |

HTTP/HTTPS servers need additional dependencies installed in the server environment:

```bash
python -m pip install "vlastudio[serve-http]>=0.2.2"
```

Update the address in both scripts. For HTTPS, set `ILSTD_SSL_KEYFILE` and
`ILSTD_SSL_CERTFILE` to the certificate paths. SHM does not use the network;
both processes must have access to the same shared memory.

TCP transport uses Python pickle and should run on a trusted network or locally.
For untrusted networks, use HTTPS or a trusted tunnel.

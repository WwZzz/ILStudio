# Train
```shell
cd /path/to/ILStudio
python train.py -p act_libero -t libero -c default -o ckpt/act_libero_object
```

# Eval
### Start policy as server
```shell
python start_server_policy -m ckpt/act_libero_object --port 5000
```

### Evaluate the policy in simulation
```shell
python eval_sim.py -m localhost:5000 -e libero --use_spawn
```
# NavLock-HY Dataset Notes

NavLock-HY is a real fixed-infrastructure navigation-lock dataset collected at
Huaiyin No. 3 Navigation Lock in Huai'an, Jiangsu Province, China.

The public code expects a nuScenes-style data root with additional
navigation-lock-specific labels. The dataset is not distributed in this
repository.

## Expected Root

```text
data/
├── v1.0-trainval/
│   ├── sample.json
│   ├── sample_data.json
│   ├── sensor.json
│   ├── calibrated_sensor.json
│   ├── sample_annotation.json
│   └── ...
├── samples/
├── sweeps/
├── splits/
│   ├── train_scenes.txt
│   ├── val_scenes.txt
│   └── test_scenes.txt
├── 2d_annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
└── navlock_sequences/
```

## Sensors

The paper setting uses:

- eight fixed camera channels;
- four fixed LiDAR channels;
- online lock-state signals for gate/water state and water levels.

## Lock-Specific Labels

In addition to standard perception labels, NavLock-HY includes labels for:

- upper/lower gate state;
- chamber, upstream, and downstream water levels;
- water-transition phase;
- ideal berth layout and berth occupancy;
- vessel intention and vessel motion flow;
- valid/invalid action masks and invalid-action reasons;
- observed operation;
- future gate/water and vessel-rollout targets.

## Rebuilding Sequence Files

```bash
PYTHONPATH=. python tools/build_navlock_sequences.py \
  --data-root data \
  --out-dir data/navlock_sequences
```

# Robot Assets

This directory contains simulation assets grouped by robot. Keep source, base, and physics USD
variants together so their references remain portable.

## Active SO-101 asset

```text
robots/so101/so101_new_calib_physics.usd
```

- `*_base.usd`: referenced geometry and articulation source.
- `*_physics.usd`: physics-ready layer used by the simulator.
- Unsuffixed `.usd`: composed or original asset.

Do not edit collected dependency paths without checking their USD references. Large generated
assets and licensed third-party files should retain their original notices.
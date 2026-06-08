# Third-party Code

This repository vendors modified third-party components required to reproduce ECA.

## LAVIS

`third_party/LAVIS/` is a minimal patched snapshot of Salesforce LAVIS. We keep only the Python package, install metadata, and license files needed by ECA. The original documentation, examples, project templates, tests, assets, and git history are omitted to keep the repository lightweight.

The patch in `third_party/patches/lavis.patch` records the ECA-specific changes, including path handling, local cache defaults, and the default-disabled BLIP-Diffusion model registration.

## AdapterHub adapters

`third_party/adapters-0.2.2/` is a patched copy of AdapterHub `adapters==0.2.2`. The patch in `third_party/patches/adapters-0.2.2.patch` records the ECA-specific adapter fusion and compatibility changes.

## LLaVA and InternVL

`third_party/LLaVA/` and `third_party/InternVL/` are compatible source snapshots used by the ECA LLaVA-style and InternVL-2.5 experiments. They are kept under `third_party/` to separate upstream backbone code from ECA model code.

Model weights are not part of the third-party source snapshots. Place them under the top-level `checkpoints/LLaVA/` and `checkpoints/InternVL/` folders as described in the main README.

Install vendored Python packages from this repository with `--no-deps` so that pip does not overwrite the tested Transformers stack.

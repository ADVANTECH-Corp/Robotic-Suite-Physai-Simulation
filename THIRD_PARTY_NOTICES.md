# Third-Party Notices

This project bundles source code from several third parties. Each component
keeps its own license; nothing here overrides those terms.

Full license texts live in [`LICENSES/`](LICENSES/). Per-file copyright and
SPDX headers are authoritative for individual files.

The root [`LICENSE`](LICENSE) (Apache-2.0) covers this project's own original
files — `Dockerfile`, `entrypoint.sh`, `requirements.txt`, `README.md`, the
Isaac Lab modification listed below, and the SAM 3 generated scene objects. It
does not change the terms of any bundled component below.

This software contains source code provided by NVIDIA Corporation.

---

## Not redistributed: NVIDIA Isaac Sim

The container is built `FROM nvcr.io/nvidia/isaac-sim`. That base image is
**not** redistributed with this repository — you pull it yourself from NVIDIA
NGC and accept NVIDIA's license directly (see README section 2). Isaac Sim's
terms are in `/isaac-sim/LICENSE.txt` inside that image, and they prohibit
redistributing the Kit itself.

Running the built image therefore requires your own valid entitlement to use
Isaac Sim.

---

## Bundled components

| Component | Path | License | Copyright |
|---|---|---|---|
| Isaac Lab | `simulation/third_party/IsaacLab/` | BSD-3-Clause | The Isaac Lab Project Developers |
| Isaac Lab Mimic | `simulation/third_party/IsaacLab/` (mimic parts) | Apache-2.0 | see `simulation/third_party/IsaacLab/LICENSE-mimic` |
| Eclipse Cyclone DDS | `simulation/third_party/cyclonedds/` | EPL-2.0 OR EDL-1.0 (BSD-3-Clause) | Eclipse Cyclone DDS project — see its `NOTICE.md` |
| robotis_dds_python | `simulation/third_party/robotis_dds_python/` | Apache-2.0 | ROBOTIS CO., LTD. |
| robotis_lab | `simulation/source/robotis_lab/` | Apache-2.0 **and** BSD-3-Clause (mixed, per file) | ROBOTIS CO., LTD. / The Isaac Lab Project Developers |
| Simulation scripts | `simulation/scripts/` | Apache-2.0 **and** BSD-3-Clause (mixed, per file) | ROBOTIS CO., LTD. / The Isaac Lab Project Developers |
| LeIsaac | `simulation/source/robotis_lab/data/robots/SO101/` (assets only) | Apache-2.0 | Lightwheel and the LeIsaac Project Developers — <https://github.com/LightwheelAI/leisaac> |
| LeRobot | installed via pip at image build (`lerobot==0.3.3`) | Apache-2.0 | The HuggingFace Inc. team |
| NVIDIA Omniverse Local Streaming Sample | `isaacsim-viewer/` | MIT (source files), distributed under the NVIDIA Omniverse License Agreement, which expressly permits distributing components identified as samples | NVIDIA CORPORATION & AFFILIATES |
| `@nvidia/omniverse-webrtc-streaming-library` | npm dependency of `isaacsim-viewer/`, fetched at build time | See its own `LICENSE.txt` — reserves all rights; it ships as a declared dependency of NVIDIA's own sample | NVIDIA CORPORATION |

### Mixed-license directories

`simulation/scripts/` and `simulation/source/robotis_lab/` contain files under
two different licenses. The split follows the per-file header, roughly:

- `scripts/imitation_learning/`, `scripts/reinforcement_learning/` — BSD-3-Clause, derived from Isaac Lab
- `scripts/sim2real/`, `scripts/tools/` — Apache-2.0, ROBOTIS
- `source/robotis_lab/` — mostly Apache-2.0 (ROBOTIS), with Isaac Lab extension-template files under BSD-3-Clause

Always check the header of the specific file.

---

## Modifications to Isaac Lab

`simulation/third_party/IsaacLab/` is a vendored copy of Isaac Lab **2.3.0**
(isaaclab package 0.47.2), with one local addition:

- `source/isaaclab/isaaclab/envs/manager_based_env.py` — added
  `_wait_for_camera_rgb_frames()` and its call site, so camera sensors are
  guaranteed to produce valid RGB frames before `ObservationManager` locks
  observation term shapes.

BSD-3-Clause does not require marking modified files; this is recorded for
traceability.

---

## Modifications to ROBOTIS material

The following material originates from ROBOTIS and **has been modified** in this
project:

- `simulation/source/robotis_lab/` — a copy of ROBOTIS's `robotis_lab`
  (now [`cyclo_lab`](https://github.com/ROBOTIS-GIT/cyclo_lab)). The object
  configurations under `robotis_lab/assets/object/` were modified; each carries
  a per-file modification notice.
- `simulation/source/robotis_lab/data/robots/OMX/` and `.../robots/OMY/` —
  produced by converting ROBOTIS **xacro** source into USD with the Isaac Sim
  importer, then further modified. Isaac Sim was only the conversion tool here —
  no Omniverse asset library content is involved, so NVIDIA's asset distribution
  terms do not apply to them.

These are derivative works of ROBOTIS material and stay under Apache-2.0.
Apache-2.0 section 4(b) requires modified files to carry a prominent notice
stating they were changed; this section serves as that notice.

`simulation/source/robotis_lab/data/robots/FFW/` and the remaining files under
`data/object/` also originate from ROBOTIS but are **unmodified**, so no 4(b)
notice is required for them.

---

## USD assets

USD content under `simulation/source/robotis_lab/data/`:

| Path | Origin | License |
|---|---|---|
| `robots/OMX/`, `robots/OMY/` | Converted from ROBOTIS xacro via the Isaac Sim importer, then modified — see the section above | Apache-2.0 (ROBOTIS) |
| `robots/FFW/` | ROBOTIS, unmodified | Apache-2.0 (ROBOTIS) |
| `robots/SO101/` | Taken from [LeIsaac](https://github.com/LightwheelAI/leisaac) | Apache-2.0 (Lightwheel / LeIsaac Project Developers) |
| `object/` (most files) | Generated with Meta's [SAM 3](https://ai.meta.com/research/sam3/) | Owned by this project — see below |
| `object/` (remaining files) | ROBOTIS, unmodified | Apache-2.0 (ROBOTIS) |

### Objects generated with SAM 3

Most scene objects under `data/object/` were reconstructed using Meta's Segment
Anything Model 3, which is distributed under the [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
(a custom Meta license, not an OSI-approved open source license).

Three points make this workable:

- **Ownership of output.** SAM License section 5(a) states that, as between you
  and Meta, you own the derivative works and modifications you make from the SAM
  Materials. The generated meshes are therefore this project's to distribute.
- **No SAM Materials are redistributed.** The license's redistribution
  obligation (section 1(b)(i) — ship a copy of the Agreement alongside) attaches
  to the SAM Materials themselves: model code, trained weights, and
  inference/training code. None of those are in this repository, only model
  output.
- **Source imagery is first-party.** The photographs used for reconstruction
  were taken by this project's author, so no third-party image rights are
  implicated in the generated meshes.

---

## License headers

Every source file in `simulation/scripts/` and `simulation/source/robotis_lab/`
carries a license header, except four zero-byte `__init__.py` namespace markers
(`assets/`, `real_world_tasks/manager_based/`, `simulation_tasks/manager_based/`,
`simulation_tasks/manager_based/FFW_BG2/reach/agents/`). Empty files contain no
copyrightable expression and need no header.

Configuration and data files (`.json`, `.yaml`, `.toml`, `.xml`, `.obj`, `.usd`)
do not carry headers, following the convention of the upstream projects.

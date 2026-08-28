# Robotis Simulation Container

A container image built on NVIDIA Isaac Sim, bundling Isaac Lab, the `robotis_lab`
simulation environments, CycloneDDS and LeRobot. Built with plain `docker build`.

---

## 1. Requirements

| Item | Requirement |
|---|---|
| GPU | NVIDIA RTX series (required by Isaac Sim; RTX 3070 / 8 GB VRAM or better recommended) |
| Driver | NVIDIA Driver 535 or newer |
| Docker | Docker Engine 20.10+ with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed |
| Disk | ~60 GB during the build (final image is ~30 GB) |
| Network | Outbound access at build time (PyPI, GitHub, nvcr.io) |

Verify the GPU is visible inside a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## 2. Get access to the Isaac Sim base image

The base image comes from NVIDIA NGC, so log in first:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API Key>   (free at https://ngc.nvidia.com)
```

## 3. Build

```bash
cd Physical-AI-Tool-Sim
docker build -t pait/simulation:latest .
```

The first build takes roughly 40–90 minutes, depending on network and CPU
(CycloneDDS is compiled from source).

Optional build args:

| Build arg | Default | Description |
|---|---|---|
| `ISAACSIM_VERSION_ARG` | `5.1.0` | Isaac Sim version |
| `USERNAME` / `USER_UID` / `USER_GID` | `pait` / `1000` / `1000` | Non-root user inside the container. Match your host's `id -u` / `id -g` to avoid permission problems on mounted directories. |
| `ROBOTISLAB_PATH_ARG` | `/workspace/robotic_suite` | Path of the project inside the container |

Example:

```bash
docker build \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -t pait/simulation:latest .
```

## 4. Run

```bash
xhost +local:docker          # allow the container to open GUI windows

docker run -it --rm \
  --gpus all \
  --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/datasets:/workspace/robotic_suite/datasets \
  -v $(pwd)/logs:/workspace/robotic_suite/logs \
  pait/simulation:latest
```

Without a display (pure training / headless), drop `DISPLAY` and the X11 mount,
and pass `--headless` to the scripts.

## 5. Environment inside the container

| Command / variable | Description |
|---|---|
| `isaaclab` | Isaac Lab entry script (`isaaclab.sh`) |
| `python` / `pip` | Aliased to the Isaac Sim built-in Python (**not** the system `python3`) |
| `lerobot-python` / `lerobot-pip` | Dedicated LeRobot virtualenv, separate from the Isaac Sim Python to avoid dependency conflicts |
| `$ROBOTISLAB_PATH` | `/workspace/robotic_suite` (working directory) |
| `$ISAACLAB_PATH` | `$ROBOTISLAB_PATH/third_party/IsaacLab` |
| `$CYCLONEDDS_HOME` | CycloneDDS install path (used for DDS communication) |
| `ROS_DOMAIN_ID` | Defaults to `30`; must match external DDS nodes |

Examples:

```bash
# Reinforcement learning training
isaaclab -p scripts/reinforcement_learning/... --headless

# Teleoperated demo recording
isaaclab -p scripts/imitation_learning/isaaclab_recorder/record_demos.py --help
```

## 6. Directory layout

```
.
├── Dockerfile              build definition
├── entrypoint.sh           recreates the _isaac_sim symlink at startup
├── requirements.txt        pinned deps for the Isaac Sim Python (pip freeze)
├── .dockerignore
├── datasets/               mount point: datasets (never baked into the image)
├── logs/                   mount point: training output
├── isaacsim-viewer/        separate image: web viewer for the WebRTC stream
└── simulation/
    ├── scripts/            training / recording / sim2real scripts
    ├── source/robotis_lab/ simulation environments and USD assets
    └── third_party/
        ├── IsaacLab/           Isaac Lab framework
        ├── cyclonedds/         DDS C library (compiled during the build)
        └── robotis_dds_python/ DDS Python bindings
```

## 7. Troubleshooting

**Build seems stuck at `cmake --build`**: compiling CycloneDDS just takes a while.
Nothing to configure — wait it out.

**`Failed to create window` / no display at runtime**: make sure you ran
`xhost +local:docker` and that `DISPLAY` is passed through. Over a remote
connection, use headless mode or Isaac Sim WebRTC streaming instead.

**Permission errors on mounted directories**: the container runs as UID 1000. If
your host account is not 1000, build with
`--build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)`.

**Very slow first startup**: Isaac Sim compiles its shader cache. To keep the
cache between runs, mount named volumes, e.g.
`-v isaac-cache-kit:/isaac-sim/kit/cache -v isaac-cache-ov:/home/pait/.cache/ov`.

## 8. Isaac Sim Viewer (optional)

Robotic Suite's `/simulate` pages embed a web viewer that displays the Isaac Sim
WebRTC stream. It is a separate, much smaller image:

```bash
docker build -t pait/isaacsim-viewer:latest ./isaacsim-viewer
```

Takes a couple of minutes. The build pulls `@nvidia/omniverse-webrtc-streaming-library`
from NVIDIA's registry, so it needs outbound access to `edge.urm.nvidia.com`.

Run it alongside the simulation container:

```bash
docker run -d --restart always -p 5173:5173 \
  --name pait_isaacsim-viewer \
  pait/isaacsim-viewer:latest
```

Then open Robotic Suite's `/simulate` page, or browse to
`http://<host>:5173?server=<host>` directly.

**Build this whenever you build the simulation image.** Without it the simulate
pages load but the viewport stays blank — the page has no way to tell you the
viewer is missing.

## 9. Licensing

This project bundles third-party code under several licenses — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the component-by-component
breakdown, with full texts in [`LICENSES/`](LICENSES/). Both are also copied
into the image at `$ROBOTISLAB_PATH`.

The NVIDIA Isaac Sim base image is **not** redistributed here: you pull it from
NGC yourself (section 2) under NVIDIA's own terms, and running the built image
requires your own valid entitlement to use Isaac Sim.

# ENPM690_Humanoid_PPO_Balancing
ENPM690 Final Project: Self-Balancing Humanoid Robot Using Reinforcement Learning and PPO

---
Group Members:
- Rey Roque (120498626)
- Kyle DeGuzman ()
---

# Docker Environment Setup

This project uses Docker for environment consistency across development machines.

Supported host platforms:

- **WSL2 (Windows + Ubuntu)**
- **Native Linux**

---

# Project Structure

```bash
docker/
├── Dockerfile
├── compose.yaml
├── compose.wslg.yaml
└── compose.linux-gui.yaml
```

---

# Requirements

Install the following before proceeding:

### Docker
- Docker Desktop (Windows/WSL2)
- Docker Engine + Docker Compose Plugin (Linux)

### Verify Installation

```bash
docker --version
docker compose version
```

---

# Building the Container

From the repository root:

```bash
docker compose -f docker/compose.yaml build
```

This builds the base development container.

---

# Running the Container (Headless)

This will launch the container with no GUI capabilities. Used during training, doesn't waste resources on rendering.

From the repository root:

```bash
docker compose -f docker/compose.yaml run --rm dev
```

---

# Running With GUI Viewer

## WSL2 / Windows

This will launch the container with GUI capabilities. Used during development for debugging and recording videos, wastes a lot of resources on rendering.

Use the WSLg GUI-enabled compose overlay:

```bash
docker compose \
-f docker/compose.yaml \
-f docker/compose.wslg.yaml \
run --rm dev
```

Requirements:

- Windows / WSL2
- WSLg enabled
- Docker desktop
---

## Linux

This will launch the container with GUI capabilities. Used during development for debugging and recording videos, wastes a lot of resources on rendering.

Use the Linux GUI compose overlay:

```bash
docker compose \
-f docker/compose.yaml \
-f docker/compose.linux-gui.yaml \
run --rm dev
```

Requirements:

- X11/Wayland desktop environment
- Docker allowed to access display server

---

# Notes

# Rebuilding the Container

If Dockerfile, dependencies, or environment configuration changes, rebuild the container.

### Standard Rebuild (Uses Docker Cache)

Recommended for most rebuilds:

```bash
docker compose -f docker/compose.yaml build
```

---

### Full Rebuild (No Cache)

Use when you change dependencies or:

- Dockerfile changes are not applying correctly
- You want a completely fresh rebuild

```bash
docker compose -f docker/compose.yaml build --no-cache
```

---

# Exiting the Container

Inside the container:

```bash
exit
```

or press:

```bash
CTRL + D
```

---
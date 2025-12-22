# Substation Extensibility Pipeline

Automated pipeline for analyzing electrical substation satellite imagery to assess extensibility potential.

## Quick Start

```bash
# Clone repository
git clone https://github.com/Daknous/substation-pipeline.git
cd full_pipeline

# Add your data
cp /path/to/substations.json input/
cp /path/to/models/* models/

# Run pipeline
docker compose up
```

## Using the CLI

The `pipeline.sh` script is a CLI that provides user-friendly execution:

```bash
# Update to latest version
./pipeline.sh pull
./pipeline.sh build

# Run full pipeline
./pipeline.sh run

# Run specific service
./pipeline.sh run service qgis_snapshots

# Check status
./pipeline.sh status

# View logs
./pipeline.sh logs          # All services
./pipeline.sh logs unet     # Specific service

# Stop pipeline
./pipeline.sh stop

# Clean outputs (careful!)
./pipeline.sh clean
```

## User Guide

### Daily Workflow

1. **Update and run with new substations:**
```bash
# Morning: pull latest code
./pipeline.sh pull
./pipeline.sh build

# Add new substations to input/substations.json
# Then run (will only process new ones)
./pipeline.sh run
```

2. **Check progress:**
```bash
./pipeline.sh status
./pipeline.sh logs
```

3. **Get results:**
```bash
ls -la output/score_results/
```

### Setting Up Remote Repository

```bash
# Set repository URL
export PIPELINE_REPO_URL=https://github.com/yourorg/substation-pipeline.git

# Initialize and push
./pipeline.sh init
git push -u origin main
```

## Pipeline Architecture

```
Input (substations.json)
    ↓
[prepare_manifest] → CSV manifest
    ↓
[osm_fetch] → Footprint geometries (resume-able)
    ↓
[qgis_snapshots] → Satellite images (skip existing)
    ↓
    ├→ [unet_infer] → Transformer segmentation (skip existing)
    │      ↓
    │   [capacity_infer] → MVA estimation
    │
    └→ [yolo_infer] → Component detection (skip existing)
           ↓
       [score] → Final extensibility scores
```

## Incremental Processing

The pipeline intelligently skips already-processed substations:
- ✅ OSM fetch: `--resume` flag
- ✅ QGIS snapshots: `--skip_existing` flag  
- ✅ U-Net inference: `--skip_existing` flag
- ✅ YOLO inference: `--skip_existing` flag

When you add new substations, only the new ones are processed!

## Troubleshooting

### Pipeline won't start
```bash
./pipeline.sh status    # Check what's missing
docker compose logs     # Check for errors
```

### Reset everything
```bash
./pipeline.sh stop
./pipeline.sh clean
./pipeline.sh run
```

### Manual service control
```bash
# Skip to specific stage
docker compose run --rm unet_infer

# Rebuild single service
docker compose build qgis_snapshots
```

## License

# Substation Extensibility Pipeline

Automated analysis of electrical substation satellite imagery to assess expansion potential using computer vision and machine learning.

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Daknous/substation-pipeline.git
cd substation-pipeline

# 2. Add your data
cp /path/to/substations.json input/
cp /path/to/models/* models/

# 3. Run pipeline
./pipeline.sh run
```

## Prerequisites

- Docker & Docker Compose
- Input data: `input/substations.json`
- Model files: `models/` (unet.pth, yolo.pt, capacity_model.joblib)
- QGIS project: `data/qgis/image_snapshots_latest.qgz` (optional)

## Usage

### Basic Commands

```bash
./pipeline.sh run         # Run complete pipeline
./pipeline.sh status      # Check pipeline status
./pipeline.sh clean       # Clean all outputs
./pipeline.sh help        # Show all commands
```

### Configuration

Create `.env` file to customize behavior:

```bash
# Skip image outputs (CSV only, saves disk space)
PIPELINE_NO_IMAGES=true

# OSM fetch settings
REQUEST_TIMEOUT_SEC=60
OSM_FETCH_MAX_TIME=300
```

### Run Modes

```bash
# Default mode (respects .env settings)
./pipeline.sh run

# Force run with all image outputs
./pipeline.sh run with-images

# Run specific service
./pipeline.sh service osm_fetch
```

## Pipeline Stages

| Stage | Purpose | Output |
|-------|---------|--------|
| `prepare_manifest` | Parse input JSON | `data/manifests/substations_manifest.csv` |
| `osm_fetch` | Fetch OSM footprints | `data/footprints/footprints.csv` |
| `qgis_snapshots` | Generate satellite images | `data/snapshots/*.png` |
| `unet_infer` | Detect transformers | `output/unet_results/transformer_detections.csv` |
| `yolo_infer` | Detect components | `output/yolo_results/yolo_summary.csv` |
| `capacity_infer` | Estimate capacity | `output/capacity_results/substations_capacity_summary.csv` |
| `score` | Calculate final scores | `output/score_results/substations_scored.csv` |

## Output Structure

```
output/
├── unet_results/           # Transformer detections
│   ├── transformer_detections.csv
│   ├── overlays/          # (if images enabled)
│   └── masks/             # (if images enabled)
├── yolo_results/           # Component detections
│   ├── yolo_summary.csv
│   ├── yolo_detections.csv
│   └── annotated/         # (if images enabled)
├── capacity_results/       # Capacity estimates
│   └── substations_capacity_summary.csv
└── score_results/          # Final scores
    └── substations_scored.csv
```

## Incremental Processing

The pipeline automatically skips already-processed substations:
- OSM fetch resumes from last successful query
- Snapshots skip existing images
- ML inference skips completed detections

Add new substations to `input/substations.json` and run again - only new items are processed.

## Troubleshooting

**Pipeline fails at OSM fetch:**
```bash
# OSM servers can be overloaded. Either:
# 1. Wait and retry
./pipeline.sh service osm_fetch

# 2. Skip OSM and use fixed buffer
mkdir -p data/footprints
echo "osm_id,footprint_wkt,found,method,note,estimated_size_m,Latitude,Longitude,Name,Voltages" > data/footprints/footprints.csv
./pipeline.sh run
```

**Check what's running:**
```bash
./pipeline.sh status
./pipeline.sh logs
```

**Reset and start fresh:**
```bash
./pipeline.sh clean
./pipeline.sh run
```

## License

[Add license information]
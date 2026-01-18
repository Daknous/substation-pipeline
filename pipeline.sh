#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
DIM='\033[2m'
NC='\033[0m' # No Color

# Detect if we're on Mac and need the override file
if [[ "$OSTYPE" == "darwin"* ]]; then
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.mac.yml"
else
    COMPOSE_FILES="-f docker-compose.yml"
fi

print_header() {
    echo ""
    echo -e "${BLUE}Substation Pipeline${NC}"
    echo -e "${DIM}──────────────────────${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}•${NC} $1"
}

check_requirements() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
}

validate_files() {
    local has_error=false
    
    echo -e "${DIM}Checking requirements...${NC}"
    
    # Check required files silently
    if [ ! -f input/substations.json ]; then
        print_error "input/substations.json missing"
        has_error=true
    fi
    
    if [ ! -f models/unet.pth ]; then
        print_error "models/unet.pth missing"
        has_error=true
    fi
    
    if [ ! -f models/yolo.pt ]; then
        print_error "models/yolo.pt missing"
        has_error=true
    fi
    
    if [ ! -f models/capacity_model.joblib ]; then
        print_error "models/capacity_model.joblib missing"
        has_error=true
    fi
    
    if [ "$has_error" = true ]; then
        echo ""
        print_error "Missing required files"
        exit 1
    else
        print_success "Ready to run"
    fi
}

build() {
    print_info "Building containers..."
    docker compose $COMPOSE_FILES build --quiet
    print_success "Build complete"
}

run() {
    local with_images="${1:-}"
    
    validate_files
    echo ""
    
    if [ "$with_images" = "with-images" ]; then
        print_info "Mode: WITH images"
        PIPELINE_NO_IMAGES=false docker compose $COMPOSE_FILES up --abort-on-container-exit
    else
        if grep -q "PIPELINE_NO_IMAGES=true" .env 2>/dev/null; then
            print_info "Mode: CSV only (no images)"
        else
            print_info "Mode: Full output (with images)"
        fi
        docker compose $COMPOSE_FILES up --abort-on-container-exit
    fi
    
    echo ""
    show_results_compact
}

run_service() {
    local service="$1"
    if [ -z "$service" ]; then
        print_error "Specify a service name"
        echo ""
        echo "Available:"
        docker compose $COMPOSE_FILES config --services | sed 's/^/  • /'
        exit 1
    fi
    
    print_info "Running: $service"
    docker compose $COMPOSE_FILES run --rm "$service"
}

status() {
    print_header
    
    # Check if results exist
    if [ -f output/score_results/substations_scored.csv ]; then
        local total=$(tail -n +2 output/score_results/substations_scored.csv | wc -l | tr -d ' ')
        print_success "Processed $total substations"
        
        # Check output mode
        if [ -d output/unet_results/overlays ] || [ -d output/yolo_results/annotated ]; then
            echo -e "${DIM}Output: Full (with images)${NC}"
        else
            echo -e "${DIM}Output: CSV only${NC}"
        fi
        
        echo ""
        echo "Results: ${DIM}output/score_results/substations_scored.csv${NC}"
    else
        print_info "No results yet"
        echo -e "${DIM}Run './pipeline.sh run' to start${NC}"
    fi
}

show_results_compact() {
    if [ -f output/score_results/substations_scored.csv ]; then
        local total=$(tail -n +2 output/score_results/substations_scored.csv | wc -l | tr -d ' ')
        print_success "Complete: $total substations processed"
        echo -e "${DIM}Results in: output/score_results/${NC}"
    else
        print_error "No results generated"
    fi
}

show_results() {
    if [ -f output/score_results/substations_scored.csv ]; then
        local total=$(tail -n +2 output/score_results/substations_scored.csv | wc -l | tr -d ' ')
        echo ""
        echo "Results Summary"
        echo -e "${DIM}───────────────${NC}"
        echo "Substations: $total"
        echo "Output: output/score_results/substations_scored.csv"
        
        # Simple preview - just first 3 lines, key columns only
        echo ""
        echo -e "${DIM}Sample (first 3):${NC}"
        echo -e "${DIM}─────────────────${NC}"
        tail -n +1 output/score_results/substations_scored.csv | head -n 4 | \
            awk -F',' '{printf "%-12s %-8s %-12s %-15s\n", $1, $4, $6, $11}' | \
            sed '1s/^/\x1b[1m/; 1s/$/\x1b[0m/'
    else
        print_info "No results yet"
    fi
}

stop() {
    print_info "Stopping..."
    docker compose $COMPOSE_FILES down -v 2>/dev/null
    print_success "Stopped"
}

# Replace the clean() function (lines 189-197) with this improved version:

clean() {
    echo -n "Clean all outputs? (y/N): "
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Clean files but preserve .gitkeep
        find output -type f ! -name '.gitkeep' -delete 2>/dev/null || true
        find data/snapshots -type f ! -name '.gitkeep' -delete 2>/dev/null || true
        find data/footprints -type f ! -name '.gitkeep' -delete 2>/dev/null || true
        find data/manifests -type f ! -name '.gitkeep' -delete 2>/dev/null || true
        
        # Remove empty subdirectories (but not the main directories with .gitkeep)
        find output -mindepth 2 -type d -empty -delete 2>/dev/null || true
        find data -mindepth 2 -type d -empty -delete 2>/dev/null || true
        
        # Ensure .gitkeep files exist (restore if accidentally deleted)
        for dir in output/capacity_results output/score_results output/unet_results output/yolo_results \
                   data/snapshots data/footprints data/manifests; do
            mkdir -p "$dir"
            [ ! -f "$dir/.gitkeep" ] && touch "$dir/.gitkeep"
        done
        
        print_success "Cleaned (preserved .gitkeep files)"
    fi
}



logs() {
    local service="${1:-}"
    if [ -z "$service" ]; then
        docker compose $COMPOSE_FILES logs --tail=50 -f
    else
        docker compose $COMPOSE_FILES logs --tail=50 -f "$service"
    fi
}

show_help() {
    print_header
    
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  ${GREEN}run${NC}              Run pipeline"
    echo "  ${GREEN}run with-images${NC}  Run with all images"
    echo "  ${GREEN}status${NC}           Check status"
    echo "  ${GREEN}clean${NC}            Clean outputs"
    echo ""
    echo "  service <name>   Run specific service"
    echo "  logs [service]   View logs"
    echo "  stop             Stop pipeline"
    echo "  help             Show this help"
    
    echo ""
    echo -e "${DIM}Settings:${NC}"
    if [ -f .env ] && grep -q "PIPELINE_NO_IMAGES=true" .env; then
        echo -e "${DIM}• Images: Disabled (CSV only)${NC}"
    else
        echo -e "${DIM}• Images: Enabled${NC}"
    fi
}

# Main script logic
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

check_requirements

case "$1" in
    run)
        shift
        run "$@"
        ;;
    service)
        shift
        run_service "$@"
        ;;
    build)
        build
        ;;
    status)
        status
        ;;
    stop)
        stop
        ;;
    logs)
        shift
        logs "$@"
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown: $1"
        echo "Try: $0 help"
        exit 1
        ;;
esac

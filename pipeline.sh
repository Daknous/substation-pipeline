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
    
    # Check required files
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
    
    # Determine which service should be the final one
    local final_service="score"
    if [ -f "docker-compose.yml" ] && grep -q "aggregate:" docker-compose.yml; then
        final_service="aggregate"
    fi
    
    # Run the pipeline with appropriate mode
    if [ "$with_images" = "with-images" ]; then
        print_info "Mode: WITH images"
        PIPELINE_NO_IMAGES=false docker compose $COMPOSE_FILES up --exit-code-from $final_service
    else
        if grep -q "PIPELINE_NO_IMAGES=true" .env 2>/dev/null; then
            print_info "Mode: CSV only (no images)"
        else
            print_info "Mode: Full output (with images)"
        fi
        docker compose $COMPOSE_FILES up --exit-code-from $final_service
    fi
    
    # Check if pipeline succeeded
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        print_error "Pipeline failed with exit code: $exit_code"
        return $exit_code
    fi
    
    echo ""
    show_results_compact
}

run_service() {
    local service="$1"
    if [ -z "$service" ]; then
        print_error "Specify a service name"
        echo ""
        echo "Available services:"
        docker compose $COMPOSE_FILES config --services | sed 's/^/  • /'
        exit 1
    fi
    
    print_info "Running: $service"
    docker compose $COMPOSE_FILES run --rm "$service"
}

status() {
    print_header
    
    # Check for aggregated results first
    if [ -f output/aggregated_results.csv ]; then
        local total=$(tail -n +2 output/aggregated_results.csv | wc -l | tr -d ' ')
        print_success "Processed $total substations (aggregated)"
        
        # Check output mode
        if [ -d output/unet_results/overlays ] || [ -d output/yolo_results/annotated ]; then
            echo -e "${DIM}Output: Full (with images)${NC}"
        else
            echo -e "${DIM}Output: CSV only${NC}"
        fi
        
        echo ""
        echo "Results:"
        echo -e "  • ${GREEN}output/aggregated_results.csv${NC}"
        echo -e "  • ${GREEN}output/aggregated_results_simple.csv${NC}"
        
    elif [ -f output/score_results/substations_scored.csv ]; then
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
    if [ -f output/aggregated_results.csv ]; then
        local total=$(tail -n +2 output/aggregated_results.csv | wc -l | tr -d ' ')
        print_success "Complete: $total substations processed & aggregated"
        echo -e "${DIM}Results:${NC}"
        echo -e "  • ${GREEN}output/aggregated_results.csv${NC}"
        echo -e "  • ${GREEN}output/aggregated_results_simple.csv${NC}"
    elif [ -f output/score_results/substations_scored.csv ]; then
        local total=$(tail -n +2 output/score_results/substations_scored.csv | wc -l | tr -d ' ')
        print_success "Complete: $total substations processed"
        echo -e "${DIM}Results in: output/score_results/${NC}"
    else
        print_error "No results generated"
    fi
}

show_results() {
    if [ -f output/aggregated_results.csv ]; then
        local total=$(tail -n +2 output/aggregated_results.csv | wc -l | tr -d ' ')
        echo ""
        echo "Results Summary (Aggregated)"
        echo -e "${DIM}────────────────────────────${NC}"
        echo "Substations: $total"
        echo "Output: output/aggregated_results.csv"
        
        # Simple preview - first 3 lines, key columns only
        echo ""
        echo -e "${DIM}Sample (first 3):${NC}"
        echo -e "${DIM}─────────────────${NC}"
        tail -n +1 output/aggregated_results_simple.csv 2>/dev/null | head -n 4 | \
            awk -F',' '{printf "%-12s %-30s %-8s %-12s\n", $1, substr($2,1,30), $5, $8}' | \
            sed '1s/^/\x1b[1m/; 1s/$/\x1b[0m/'
            
    elif [ -f output/score_results/substations_scored.csv ]; then
        local total=$(tail -n +2 output/score_results/substations_scored.csv | wc -l | tr -d ' ')
        echo ""
        echo "Results Summary"
        echo -e "${DIM}───────────────${NC}"
        echo "Substations: $total"
        echo "Output: output/score_results/substations_scored.csv"
        
        # Simple preview - first 3 lines, key columns only
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

clean() {
    echo -n "Clean all outputs? (y/N): "
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Clean outputs
        find output -type f ! -name '.gitkeep' -delete 2>/dev/null || true
        
        # Clean snapshots
        find data/snapshots -type f ! -name '.gitkeep' -delete 2>/dev/null || true
        
        # Clean manifests
        find data/manifests -type f ! -name '.gitkeep' -delete 2>/dev/null || true
        
        # Clean footprints BUT PRESERVE footprints.csv if it exists
        if [ -f data/footprints/footprints.csv ]; then
            # Save footprints.csv temporarily
            cp data/footprints/footprints.csv /tmp/footprints_preserved.csv
            
            # Clean the directory
            find data/footprints -type f ! -name '.gitkeep' -delete 2>/dev/null || true
            
            # Restore footprints.csv
            mv /tmp/footprints_preserved.csv data/footprints/footprints.csv
            
            FOOTPRINT_COUNT=$(tail -n +2 data/footprints/footprints.csv | wc -l | tr -d ' ')
            echo "  Preserved footprints.csv ($FOOTPRINT_COUNT entries)"
        else
            # No footprints.csv to preserve
            find data/footprints -type f ! -name '.gitkeep' -delete 2>/dev/null || true
        fi
        
        # Remove empty subdirectories
        find output -mindepth 2 -type d -empty -delete 2>/dev/null || true
        find data -mindepth 2 -type d -empty -delete 2>/dev/null || true
        
        # Ensure .gitkeep files exist
        for dir in output/capacity_results output/score_results output/unet_results output/yolo_results \
                   data/snapshots data/footprints data/manifests; do
            mkdir -p "$dir"
            [ ! -f "$dir/.gitkeep" ] && touch "$dir/.gitkeep"
        done
        
        print_success "Cleaned outputs (preserved footprints.csv)"
    fi
}

clean_menu() {
    print_header
    echo "What would you like to clean?"
    echo ""
    echo "  1) Output files only (preserve all data)"
    echo "  2) Output + snapshots (preserve footprints & manifests)"
    echo "  3) Everything except footprints.csv"
    echo "  4) Everything (full clean)"
    echo "  5) Cancel"
    echo ""
    read -p "Choice [1-5]: " -n 1 -r
    echo
    
    case $REPLY in
        1)
            find output -type f ! -name '.gitkeep' -delete 2>/dev/null || true
            print_success "Cleaned output files only"
            ;;
        2)
            find output -type f ! -name '.gitkeep' -delete 2>/dev/null || true
            find data/snapshots -type f ! -name '.gitkeep' -delete 2>/dev/null || true
            print_success "Cleaned outputs and snapshots"
            ;;
        3)
            find output -type f ! -name '.gitkeep' -delete 2>/dev/null || true
            find data/snapshots -type f ! -name '.gitkeep' -delete 2>/dev/null || true
            find data/manifests -type f ! -name '.gitkeep' -delete 2>/dev/null || true
            
            # Preserve footprints.csv
            if [ -f data/footprints/footprints.csv ]; then
                find data/footprints -type f ! -name '.gitkeep' ! -name 'footprints.csv' -delete 2>/dev/null || true
                FOOTPRINT_COUNT=$(tail -n +2 data/footprints/footprints.csv | wc -l | tr -d ' ')
                print_success "Cleaned (preserved footprints.csv with $FOOTPRINT_COUNT entries)"
            else
                find data/footprints -type f ! -name '.gitkeep' -delete 2>/dev/null || true
                print_success "Cleaned"
            fi
            ;;
        4)
            # Full clean - ask for confirmation
            echo -n "Are you SURE? This will delete footprints.csv too! (y/N): "
            read -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                # Backup footprints.csv if it exists
                if [ -f data/footprints/footprints.csv ]; then
                    FOOTPRINT_COUNT=$(tail -n +2 data/footprints/footprints.csv | wc -l | tr -d ' ')
                    cp data/footprints/footprints.csv "/tmp/footprints_backup_$(date +%Y%m%d_%H%M%S).csv"
                    print_info "Backed up $FOOTPRINT_COUNT footprints to /tmp/"
                fi
                
                find output -type f ! -name '.gitkeep' -delete 2>/dev/null || true
                find data -type f ! -name '.gitkeep' -delete 2>/dev/null || true
                print_success "Full clean complete"
            else
                print_info "Cancelled"
            fi
            ;;
        *)
            print_info "Cancelled"
            ;;
    esac
    
    # Always ensure .gitkeep files exist
    for dir in output/capacity_results output/score_results output/unet_results output/yolo_results \
               data/snapshots data/footprints data/manifests; do
        mkdir -p "$dir"
        [ ! -f "$dir/.gitkeep" ] && touch "$dir/.gitkeep"
    done
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
    echo "  ${GREEN}results${NC}          Show detailed results"
    echo "  ${GREEN}clean${NC}            Clean outputs"
    echo ""
    echo "  service <name>   Run specific service"
    echo "  logs [service]   View logs"
    echo "  stop             Stop pipeline"
    echo "  build            Build containers"
    echo "  help             Show this help"
    
    echo ""
    echo -e "${DIM}Settings:${NC}"
    if [ -f .env ] && grep -q "PIPELINE_NO_IMAGES=true" .env; then
        echo -e "${DIM}• Images: Disabled (CSV only)${NC}"
    else
        echo -e "${DIM}• Images: Enabled${NC}"
    fi
    
    # Show footprints status
    if [ -f data/footprints/footprints.csv ]; then
        local footprint_count=$(tail -n +2 data/footprints/footprints.csv | wc -l | tr -d ' ')
        echo -e "${DIM}• Footprints: $footprint_count pre-fetched${NC}"
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
    results)
        show_results
        ;;
    stop)
        stop
        ;;
    logs)
        shift
        logs "$@"
        ;;
    clean)
        clean_menu
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

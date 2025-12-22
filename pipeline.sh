#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Pipeline configuration
REPO_URL="${PIPELINE_REPO_URL:-}"
BRANCH="${PIPELINE_BRANCH:-main}"

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Substation Pipeline Manager${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

check_requirements() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed"
        exit 1
    fi
    
    print_success "All requirements met"
}

pull_updates() {
    print_info "Pulling latest changes from $BRANCH..."
    
    if [ ! -d .git ]; then
        print_error "Not a git repository. Run 'init' first."
        exit 1
    fi
    
    # Stash any local changes
    if [[ -n $(git status -s) ]]; then
        print_info "Stashing local changes..."
        git stash push -m "Auto-stash before pull $(date +%Y%m%d_%H%M%S)"
    fi
    
    git pull origin "$BRANCH"
    print_success "Updated to latest version"
    
    # Show if there are stashed changes
    if git stash list | grep -q "Auto-stash"; then
        print_info "You have stashed changes. Run 'git stash pop' to restore them."
    fi
}

build_containers() {
    print_info "Building Docker containers..."
    docker compose build
    print_success "Containers built successfully"
}

run_pipeline() {
    local mode="${1:-full}"
    
    case "$mode" in
        full)
            print_info "Running full pipeline..."
            docker compose up
            ;;
        incremental)
            print_info "Running incremental pipeline (skip existing)..."
            docker compose up
            ;;
        service)
            if [ -z "$2" ]; then
                print_error "Please specify a service name"
                echo "Available services:"
                docker compose config --services | sed 's/^/  - /'
                exit 1
            fi
            print_info "Running service: $2"
            docker compose up "$2"
            ;;
        *)
            print_error "Unknown mode: $mode"
            echo "Available modes: full, incremental, service <name>"
            exit 1
            ;;
    esac
}

status() {
    print_header
    echo ""
    echo "Pipeline Status:"
    echo "----------------"
    docker compose ps
    
    echo ""
    echo "Repository Status:"
    echo "------------------"
    if [ -d .git ]; then
        echo "Branch: $(git branch --show-current)"
        echo "Last commit: $(git log -1 --oneline)"
        if [[ -n $(git status -s) ]]; then
            echo "Local changes: Yes"
        else
            echo "Local changes: No"
        fi
    else
        print_info "Not a git repository"
    fi
    
    echo ""
    echo "Data Status:"
    echo "------------"
    [ -f input/substations.json ] && print_success "Input data present" || print_info "No input data"
    [ -f models/unet.pth ] && print_success "U-Net model present" || print_info "U-Net model missing"
    [ -f models/yolo.pt ] && print_success "YOLO model present" || print_info "YOLO model missing"
    [ -f models/capacity_model.joblib ] && print_success "Capacity model present" || print_info "Capacity model missing"
    
    if [ -d output/score_results ] && [ "$(ls -A output/score_results 2>/dev/null)" ]; then
        echo ""
        echo "Latest Results:"
        echo "---------------"
        ls -lht output/score_results/*.csv 2>/dev/null | head -5
    fi
}

stop_pipeline() {
    print_info "Stopping pipeline..."
    docker compose down
    print_success "Pipeline stopped"
}

clean() {
    read -p "This will remove all output data. Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleaning output directories..."
        rm -rf output/*
        rm -rf data/snapshots/*
        rm -rf data/footprints/*
        rm -rf data/manifests/*
        touch output/.gitkeep
        touch data/.gitkeep
        print_success "Output directories cleaned"
    else
        print_info "Clean cancelled"
    fi
}

logs() {
    local service="${1:-}"
    if [ -z "$service" ]; then
        docker compose logs --tail=100 -f
    else
        docker compose logs --tail=100 -f "$service"
    fi
}

init_repo() {
    if [ -d .git ]; then
        print_info "Already a git repository"
        return
    fi
    
    print_info "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Substation extensibility pipeline"
    
    if [ -n "$REPO_URL" ]; then
        git remote add origin "$REPO_URL"
        print_success "Repository initialized with remote: $REPO_URL"
    else
        print_success "Repository initialized (no remote set)"
        print_info "Set PIPELINE_REPO_URL environment variable to add remote"
    fi
}

show_help() {
    print_header
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  init          Initialize git repository"
    echo "  pull          Pull latest changes from git"
    echo "  build         Build Docker containers"
    echo "  run [mode]    Run pipeline (full|incremental|service <name>)"
    echo "  status        Show pipeline and repository status"
    echo "  stop          Stop running pipeline"
    echo "  logs [svc]    Show logs (optionally for specific service)"
    echo "  clean         Clean output directories"
    echo "  help          Show this help message"
    echo ""
    echo "Quick start:"
    echo "  $0 init                    # Initialize repository"
    echo "  $0 pull && $0 build       # Update and build"
    echo "  $0 run                     # Run full pipeline"
    echo ""
    echo "Environment variables:"
    echo "  PIPELINE_REPO_URL   Git repository URL"
    echo "  PIPELINE_BRANCH     Git branch (default: main)"
}

# Main script logic
print_header

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

check_requirements

case "$1" in
    init)
        init_repo
        ;;
    pull)
        pull_updates
        ;;
    build)
        build_containers
        ;;
    run)
        shift
        run_pipeline "$@"
        ;;
    status)
        status
        ;;
    stop)
        stop_pipeline
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
        print_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac
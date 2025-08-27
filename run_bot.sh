#!/bin/bash

# Crypto Trading Bot - Deployment and Management Script
# This script handles deployment, startup, monitoring, and maintenance

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BOT_NAME="crypto-trading-bot"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
LOG_DIR="./logs"
STORAGE_DIR="./storage"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════╗"
    echo "║        🚀 CRYPTO TRADING BOT 🚀        ║"
    echo "║                                       ║"
    echo "║      Management & Deployment Tool     ║"
    echo "╚═══════════════════════════════════════╝"
    echo -e "${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root. Consider using a non-root user for better security."
    fi
    
    print_status "Prerequisites check completed ✅"
}

setup_environment() {
    print_status "Setting up environment..."
    
    # Create directories
    mkdir -p ${LOG_DIR}
    mkdir -p ${STORAGE_DIR}/{historical,models,backups,exports}
    mkdir -p config
    
    # Check if .env file exists
    if [[ ! -f ${ENV_FILE} ]]; then
        print_warning ".env file not found. Creating from template..."
        cp .env.example ${ENV_FILE}
        print_error "Please edit ${ENV_FILE} with your configuration before proceeding."
        exit 1
    fi
    
    # Set permissions
    chmod 755 ${LOG_DIR}
    chmod 755 ${STORAGE_DIR}
    
    print_status "Environment setup completed ✅"
}

validate_config() {
    print_status "Validating configuration..."
    
    # Check required environment variables
    required_vars=("BINANCE_API_KEY" "BINANCE_SECRET")
    
    source ${ENV_FILE}
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            print_error "Required environment variable ${var} is not set in ${ENV_FILE}"
            exit 1
        fi
    done
    
    # Validate trading mode
    if [[ "${TRADING_MODE}" != "paper" && "${TRADING_MODE}" != "live" ]]; then
        print_error "TRADING_MODE must be either 'paper' or 'live'"
        exit 1
    fi
    
    if [[ "${TRADING_MODE}" == "live" ]]; then
        print_warning "⚠️  LIVE TRADING MODE ENABLED ⚠️"
        print_warning "This will use real money. Are you sure? (y/N)"
        read -r response
        if [[ ! "${response}" =~ ^[Yy]$ ]]; then
            print_status "Switching to paper trading mode for safety."
            sed -i 's/TRADING_MODE=live/TRADING_MODE=paper/' ${ENV_FILE}
        fi
    fi
    
    print_status "Configuration validation completed ✅"
}

build_images() {
    print_status "Building Docker images..."
    docker-compose build --no-cache
    print_status "Docker images built successfully ✅"
}

start_services() {
    print_status "Starting services..."
    
    # Start core services
    docker-compose up -d postgres redis
    
    # Wait for database to be ready
    print_status "Waiting for database to be ready..."
    sleep 10
    
    # Start trading bot
    docker-compose up -d trading-bot
    
    print_status "Services started successfully ✅"
    
    # Show status
    show_status
}

stop_services() {
    print_status "Stopping services..."
    docker-compose down
    print_status "Services stopped ✅"
}

restart_services() {
    print_status "Restarting services..."
    docker-compose restart
    print_status "Services restarted ✅"
}

show_status() {
    print_status "Service Status:"
    docker-compose ps
    
    echo ""
    print_status "Recent logs:"
    docker-compose logs --tail=20 trading-bot
}

show_logs() {
    local service=${1:-trading-bot}
    local lines=${2:-50}
    
    print_status "Showing logs for ${service} (last ${lines} lines):"
    docker-compose logs --tail=${lines} --follow ${service}
}

backup_data() {
    print_status "Creating backup..."
    
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_dir="${STORAGE_DIR}/backups/backup_${timestamp}"
    
    mkdir -p ${backup_dir}
    
    # Backup database
    docker-compose exec -T postgres pg_dump -U trading_user trading_bot > ${backup_dir}/database.sql
    
    # Backup configuration
    cp -r config ${backup_dir}/
    cp ${ENV_FILE} ${backup_dir}/
    
    # Backup logs (last 7 days)
    find ${LOG_DIR} -name "*.log" -mtime -7 -exec cp {} ${backup_dir}/ \;
    
    # Create tar archive
    tar -czf ${backup_dir}.tar.gz -C ${STORAGE_DIR}/backups backup_${timestamp}
    rm -rf ${backup_dir}
    
    print_status "Backup created: ${backup_dir}.tar.gz ✅"
}

restore_data() {
    local backup_file=$1
    
    if [[ -z "${backup_file}" ]]; then
        print_error "Please specify backup file to restore"
        exit 1
    fi
    
    if [[ ! -f "${backup_file}" ]]; then
        print_error "Backup file ${backup_file} not found"
        exit 1
    fi
    
    print_warning "This will restore data from backup. Continue? (y/N)"
    read -r response
    if [[ ! "${response}" =~ ^[Yy]$ ]]; then
        print_status "Restore cancelled."
        exit 0
    fi
    
    print_status "Restoring from backup: ${backup_file}"
    
    # Extract backup
    temp_dir=$(mktemp -d)
    tar -xzf ${backup_file} -C ${temp_dir}
    
    # Stop services
    stop_services
    
    # Restore database
    docker-compose up -d postgres
    sleep 10
    docker-compose exec -T postgres psql -U trading_user -d trading_bot < ${temp_dir}/*/database.sql
    
    # Restore configuration
    cp -r ${temp_dir}/*/config/* config/
    
    # Start services
    start_services
    
    # Cleanup
    rm -rf ${temp_dir}
    
    print_status "Restore completed ✅"
}

update_bot() {
    print_status "Updating trading bot..."
    
    # Pull latest code
    git pull origin main
    
    # Rebuild and restart
    build_images
    restart_services
    
    print_status "Bot updated successfully ✅"
}

monitor_performance() {
    print_status "Performance monitoring..."
    
    while true; do
        clear
        echo -e "${BLUE}=== TRADING BOT PERFORMANCE MONITOR ===${NC}"
        echo ""
        
        # Container stats
        echo -e "${GREEN}Container Resources:${NC}"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
        
        echo ""
        
        # Service status
        echo -e "${GREEN}Service Status:${NC}"
        docker-compose ps
        
        echo ""
        
        # Recent logs
        echo -e "${GREEN}Recent Activity:${NC}"
        docker-compose logs --tail=5 trading-bot | tail -10
        
        echo ""
        echo -e "${YELLOW}Press Ctrl+C to exit monitoring${NC}"
        
        sleep 30
    done
}

health_check() {
    print_status "Performing health check..."
    
    # Check if containers are running
    if ! docker-compose ps | grep -q "Up"; then
        print_error "Some services are not running"
        return 1
    fi
    
    # Check database connection
    if ! docker-compose exec -T postgres pg_isready -U trading_user -d trading_bot &> /dev/null; then
        print_error "Database connection failed"
        return 1
    fi
    
    # Check Redis connection
    if ! docker-compose exec -T redis redis-cli ping &> /dev/null; then
        print_error "Redis connection failed"
        return 1
    fi
    
    # Check bot health endpoint (if available)
    if command -v curl &> /dev/null; then
        if curl -f http://localhost:8000/health &> /dev/null; then
            print_status "Bot health endpoint responding ✅"
        else
            print_warning "Bot health endpoint not responding"
        fi
    fi
    
    print_status "Health check completed ✅"
    return 0
}

cleanup_data() {
    print_status "Cleaning up old data..."
    
    # Remove old log files (>30 days)
    find ${LOG_DIR} -name "*.log.*" -mtime +30 -delete
    
    # Remove old backups (>90 days)
    find ${STORAGE_DIR}/backups -name "*.tar.gz" -mtime +90 -delete
    
    # Clean Docker system
    docker system prune -f
    
    print_status "Cleanup completed ✅"
}

install_monitoring() {
    print_status "Installing monitoring stack..."
    
    # Start monitoring services
    docker-compose --profile monitoring up -d grafana influxdb
    
    print_status "Monitoring stack installed ✅"
    print_status "Grafana available at: http://localhost:3000 (admin/admin_password)"
    print_status "InfluxDB available at: http://localhost:8086"
}

emergency_stop() {
    print_error "EMERGENCY STOP INITIATED"
    
    # Stop all containers immediately
    docker-compose kill
    
    # If in live trading, try to close positions (requires API access)
    if [[ "${TRADING_MODE}" == "live" ]]; then
        print_warning "Live trading detected. Manual position closure may be required."
    fi
    
    print_error "All services stopped. Check logs for details."
}

show_help() {
    echo "Crypto Trading Bot Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start          Start all services"
    echo "  stop           Stop all services"
    echo "  restart        Restart all services"
    echo "  status         Show service status"
    echo "  logs [service] Show logs (default: trading-bot)"
    echo "  build          Build Docker images"
    echo "  backup         Create data backup"
    echo "  restore <file> Restore from backup"
    echo "  update         Update bot code and restart"
    echo "  monitor        Monitor performance (real-time)"
    echo "  health         Perform health check"
    echo "  cleanup        Clean old data and Docker images"
    echo "  monitoring     Install monitoring stack"
    echo "  emergency      Emergency stop (kill all containers)"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start the trading bot"
    echo "  $0 logs trading-bot         # Show trading bot logs"
    echo "  $0 backup                   # Create backup"
    echo "  $0 restore backup.tar.gz    # Restore from backup"
}

# Main execution
main() {
    print_header
    
    case "${1:-help}" in
        "start")
            check_prerequisites
            setup_environment
            validate_config
            build_images
            start_services
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            restart_services
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "$2" "$3"
            ;;
        "build")
            check_prerequisites
            build_images
            ;;
        "backup")
            backup_data
            ;;
        "restore")
            restore_data "$2"
            ;;
        "update")
            update_bot
            ;;
        "monitor")
            monitor_performance
            ;;
        "health")
            health_check
            ;;
        "cleanup")
            cleanup_data
            ;;
        "monitoring")
            install_monitoring
            ;;
        "emergency")
            emergency_stop
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Trap SIGINT and SIGTERM for graceful shutdown
trap emergency_stop SIGINT SIGTERM

# Run main function
main "$@"
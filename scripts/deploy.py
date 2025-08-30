"""
Deployment script for automating bot deployment to production environment.
"""

import os
import sys
import json
import subprocess
import datetime
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import docker
import paramiko
from scp import SCPClient

logger = logging.getLogger(__name__)

class DeploymentManager:
    """Manages deployment of trading bot to various environments"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.docker_client = None
        self.ssh_client = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load deployment configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    async def deploy_local(self) -> bool:
        """Deploy bot locally using Docker"""
        try:
            logger.info("Starting local deployment...")
            
            # Initialize Docker client
            self.docker_client = docker.from_env()
            
            # Build Docker image
            logger.info("Building Docker image...")
            image, logs = self.docker_client.images.build(
                path=".",
                tag="crypto-trading-bot:latest",
                rm=True
            )
            
            for log in logs:
                if 'stream' in log:
                    print(log['stream'].strip())
            
            # Stop existing containers
            await self._stop_existing_containers()
            
            # Run docker-compose
            logger.info("Starting services with docker-compose...")
            result = subprocess.run(
                ["docker-compose", "up", "-d"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Local deployment successful")
                return True
            else:
                logger.error(f"Docker-compose failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Local deployment failed: {e}")
            return False
    
    async def deploy_vps(self, host: str, username: str, key_file: str) -> bool:
        """Deploy bot to VPS"""
        try:
            logger.info(f"Starting VPS deployment to {host}...")
            
            # Setup SSH connection
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(
                hostname=host,
                username=username,
                key_filename=key_file
            )
            
            # Create deployment directory
            remote_dir = self.config.get('remote_directory', '/opt/crypto-trading-bot')
            await self._execute_ssh_command(f"mkdir -p {remote_dir}")
            
            # Upload files
            logger.info("Uploading project files...")
            await self._upload_project_files(remote_dir)
            
            # Setup environment
            logger.info("Setting up environment...")
            await self._setup_remote_environment(remote_dir)
            
            # Deploy application
            logger.info("Deploying application...")
            await self._deploy_remote_application(remote_dir)
            
            # Start services
            logger.info("Starting services...")
            await self._start_remote_services(remote_dir)
            
            logger.info("VPS deployment successful")
            return True
            
        except Exception as e:
            logger.error(f"VPS deployment failed: {e}")
            return False
        finally:
            if self.ssh_client:
                self.ssh_client.close()
    
    async def _stop_existing_containers(self):
        """Stop existing Docker containers"""
        try:
            containers = self.docker_client.containers.list(
                filters={"label": "crypto-trading-bot"}
            )
            
            for container in containers:
                logger.info(f"Stopping container: {container.name}")
                container.stop()
                container.remove()
                
        except Exception as e:
            logger.warning(f"Failed to stop existing containers: {e}")
    
    async def _execute_ssh_command(self, command: str) -> tuple:
        """Execute SSH command and return stdout, stderr"""
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            stdout_text = stdout.read().decode()
            stderr_text = stderr.read().decode()
            return stdout_text, stderr_text
        except Exception as e:
            logger.error(f"SSH command failed: {e}")
            return "", str(e)
    
    async def _upload_project_files(self, remote_dir: str):
        """Upload project files to VPS"""
        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                # Upload essential files
                files_to_upload = [
                    'main.py',
                    'requirements.txt',
                    'Dockerfile',
                    'docker-compose.yml',
                    '.env.example',
                    'run_bot.sh'
                ]
                
                # Upload directories
                dirs_to_upload = [
                    'core/',
                    'data/',
                    'strategies/',
                    'risk/',
                    'utils/',
                    'config/',
                    'monitoring/',
                    'scripts/'
                ]
                
                # Upload files
                for file in files_to_upload:
                    if os.path.exists(file):
                        scp.put(file, f"{remote_dir}/{file}")
                        logger.info(f"Uploaded {file}")
                
                # Upload directories recursively
                for dir_path in dirs_to_upload:
                    if os.path.exists(dir_path):
                        scp.put(dir_path, remote_dir, recursive=True)
                        logger.info(f"Uploaded {dir_path}")
                
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise
    
    async def _setup_remote_environment(self, remote_dir: str):
        """Setup remote environment"""
        commands = [
            # Install Docker if not present
            "curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh || true",
            
            # Install Docker Compose
            'sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose',
            "sudo chmod +x /usr/local/bin/docker-compose",
            
            # Create necessary directories
            f"cd {remote_dir} && mkdir -p logs storage/historical storage/models storage/backups storage/exports",
            
            # Set permissions
            f"cd {remote_dir} && chmod +x run_bot.sh",
            f"cd {remote_dir} && chmod +x scripts/*.py",
            
            # Copy environment file
            f"cd {remote_dir} && cp .env.example .env",
        ]
        
        for command in commands:
            stdout, stderr = await self._execute_ssh_command(command)
            if stderr and "warning" not in stderr.lower():
                logger.warning(f"Command warning: {stderr}")
    
    async def _deploy_remote_application(self, remote_dir: str):
        """Deploy application on remote server"""
        try:
            # Build and start containers
            commands = [
                f"cd {remote_dir}",
                "docker-compose down --remove-orphans",
                "docker-compose build --no-cache",
                "docker-compose up -d"
            ]
            
            for command in commands:
                stdout, stderr = await self._execute_ssh_command(command)
                logger.info(f"Command output: {stdout}")
                if stderr:
                    logger.warning(f"Command stderr: {stderr}")
            
        except Exception as e:
            logger.error(f"Remote deployment failed: {e}")
            raise
    
    async def _start_remote_services(self, remote_dir: str):
        """Start services on remote server"""
        try:
            # Start main services
            stdout, stderr = await self._execute_ssh_command(
                f"cd {remote_dir} && ./run_bot.sh start"
            )
            logger.info(f"Services started: {stdout}")
            
            # Check service status
            stdout, stderr = await self._execute_ssh_command(
                f"cd {remote_dir} && ./run_bot.sh status"
            )
            logger.info(f"Service status: {stdout}")
            
            # Setup monitoring if configured
            if self.config.get('monitoring', {}).get('enabled', False):
                await self._setup_monitoring(remote_dir)
            
        except Exception as e:
            logger.error(f"Failed to start remote services: {e}")
            raise
    
    async def _setup_monitoring(self, remote_dir: str):
        """Setup monitoring stack"""
        try:
            monitoring_commands = [
                f"cd {remote_dir} && ./run_bot.sh monitoring",
                "sleep 10",  # Wait for services to start
                f"cd {remote_dir} && ./run_bot.sh health"
            ]
            
            for command in monitoring_commands:
                stdout, stderr = await self._execute_ssh_command(command)
                logger.info(f"Monitoring setup: {stdout}")
                
        except Exception as e:
            logger.warning(f"Monitoring setup failed: {e}")
    
    async def deploy_aws(self) -> bool:
        """Deploy to AWS using ECS or EC2"""
        try:
            logger.info("Starting AWS deployment...")
            
            aws_config = self.config.get('aws', {})
            if not aws_config:
                logger.error("AWS configuration not found")
                return False
            
            # This would require boto3 and proper AWS credentials
            # For now, we'll provide a basic framework
            
            deployment_type = aws_config.get('deployment_type', 'ec2')
            
            if deployment_type == 'ecs':
                return await self._deploy_ecs()
            elif deployment_type == 'ec2':
                return await self._deploy_ec2()
            else:
                logger.error(f"Unsupported AWS deployment type: {deployment_type}")
                return False
                
        except Exception as e:
            logger.error(f"AWS deployment failed: {e}")
            return False
    
    async def _deploy_ecs(self) -> bool:
        """Deploy using AWS ECS"""
        # This would require proper AWS SDK implementation
        logger.warning("ECS deployment not fully implemented")
        return False
    
    async def _deploy_ec2(self) -> bool:
        """Deploy to AWS EC2 instance"""
        try:
            aws_config = self.config.get('aws', {})
            instance_ip = aws_config.get('instance_ip')
            key_file = aws_config.get('key_file')
            username = aws_config.get('username', 'ubuntu')
            
            if not all([instance_ip, key_file]):
                logger.error("AWS EC2 configuration incomplete")
                return False
            
            # Use VPS deployment method for EC2
            return await self.deploy_vps(instance_ip, username, key_file)
            
        except Exception as e:
            logger.error(f"EC2 deployment failed: {e}")
            return False
    
    async def validate_deployment(self, host: str = None) -> Dict[str, Any]:
        """Validate deployment by checking service health"""
        try:
            if host:
                # Remote validation
                return await self._validate_remote_deployment(host)
            else:
                # Local validation
                return await self._validate_local_deployment()
                
        except Exception as e:
            logger.error(f"Deployment validation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _validate_local_deployment(self) -> Dict[str, Any]:
        """Validate local deployment"""
        try:
            if not self.docker_client:
                self.docker_client = docker.from_env()
            
            # Check containers
            containers = self.docker_client.containers.list(
                filters={"label": "crypto-trading-bot"}
            )
            
            container_status = {}
            for container in containers:
                container_status[container.name] = {
                    'status': container.status,
                    'health': container.attrs.get('State', {}).get('Health', {}).get('Status', 'unknown')
                }
            
            # Check if main bot is running
            bot_running = any(
                c.status == 'running' for c in containers 
                if 'trading-bot' in c.name
            )
            
            # Test API endpoint
            api_healthy = await self._test_api_endpoint('http://localhost:8000/health')
            
            validation_result = {
                'status': 'healthy' if bot_running and api_healthy else 'unhealthy',
                'containers': container_status,
                'api_healthy': api_healthy,
                'timestamp': datetime.now().isoformat()
            }
            
            return validation_result
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    async def _validate_remote_deployment(self, host: str) -> Dict[str, Any]:
        """Validate remote deployment"""
        try:
            # Test API endpoint
            api_healthy = await self._test_api_endpoint(f'http://{host}:8000/health')
            
            # Check service status via SSH
            stdout, stderr = await self._execute_ssh_command(
                f"cd {self.config.get('remote_directory', '/opt/crypto-trading-bot')} && ./run_bot.sh status"
            )
            
            return {
                'status': 'healthy' if api_healthy and 'running' in stdout.lower() else 'unhealthy',
                'api_healthy': api_healthy,
                'service_status': stdout,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    async def _test_api_endpoint(self, url: str) -> bool:
        """Test if API endpoint is responding"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    async def rollback_deployment(self, host: str = None) -> bool:
        """Rollback to previous deployment"""
        try:
            logger.info("Starting deployment rollback...")
            
            if host:
                # Remote rollback
                remote_dir = self.config.get('remote_directory', '/opt/crypto-trading-bot')
                commands = [
                    f"cd {remote_dir}",
                    "git checkout HEAD~1",  # Go back one commit
                    "docker-compose down",
                    "docker-compose build",
                    "docker-compose up -d"
                ]
                
                for command in commands:
                    stdout, stderr = await self._execute_ssh_command(command)
                    logger.info(f"Rollback command: {command} -> {stdout}")
            else:
                # Local rollback
                subprocess.run(["git", "checkout", "HEAD~1"])
                subprocess.run(["docker-compose", "down"])
                subprocess.run(["docker-compose", "build"])
                subprocess.run(["docker-compose", "up", "-d"])
            
            logger.info("Rollback completed")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def generate_deployment_config(self, environment: str) -> Dict[str, Any]:
        """Generate deployment configuration for different environments"""
        base_config = {
            'app_name': 'crypto-trading-bot',
            'version': '1.0.0',
            'environment': environment
        }
        
        if environment == 'development':
            return {
                **base_config,
                'trading_mode': 'paper',
                'log_level': 'DEBUG',
                'database_url': 'sqlite:///data/trading_bot_dev.db',
                'redis_url': 'redis://localhost:6379/0'
            }
        
        elif environment == 'staging':
            return {
                **base_config,
                'trading_mode': 'paper',
                'log_level': 'INFO',
                'database_url': 'postgresql://user:pass@staging-db:5432/trading_bot',
                'redis_url': 'redis://staging-redis:6379/0'
            }
        
        elif environment == 'production':
            return {
                **base_config,
                'trading_mode': 'live',
                'log_level': 'INFO',
                'database_url': 'postgresql://user:pass@prod-db:5432/trading_bot',
                'redis_url': 'redis://prod-redis:6379/0',
                'monitoring_enabled': True,
                'backup_enabled': True
            }
        
        return base_config

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy crypto trading bot')
    parser.add_argument('--config', required=True, help='Deployment config file')
    parser.add_argument('--environment', choices=['local', 'vps', 'aws'], default='local', help='Deployment environment')
    parser.add_argument('--host', help='VPS host IP/domain')
    parser.add_argument('--username', default='root', help='SSH username')
    parser.add_argument('--key-file', help='SSH private key file')
    parser.add_argument('--validate', action='store_true', help='Only validate existing deployment')
    parser.add_argument('--rollback', action='store_true', help='Rollback deployment')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    async def run_deployment():
        try:
            deployer = DeploymentManager(args.config)
            
            if args.rollback:
                success = await deployer.rollback_deployment(args.host)
                print(f"Rollback {'successful' if success else 'failed'}")
                return
            
            if args.validate:
                result = await deployer.validate_deployment(args.host)
                print(f"Validation result: {json.dumps(result, indent=2)}")
                return
            
            # Deploy based on environment
            if args.environment == 'local':
                success = await deployer.deploy_local()
            elif args.environment == 'vps':
                if not all([args.host, args.key_file]):
                    print("VPS deployment requires --host and --key-file")
                    return
                success = await deployer.deploy_vps(args.host, args.username, args.key_file)
            elif args.environment == 'aws':
                success = await deployer.deploy_aws()
            
            if success:
                print(f"Deployment to {args.environment} successful!")
                
                # Validate deployment
                validation = await deployer.validate_deployment(args.host)
                print(f"Validation: {validation['status']}")
            else:
                print(f"Deployment to {args.environment} failed!")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            sys.exit(1)
    
    # Run deployment
    asyncio.run(run_deployment())

if __name__ == "__main__":
    main()
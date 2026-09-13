// Chat-persistence database for the RAG assistant.
//
// Holds ONLY LangGraph's checkpoint_* tables - the conversation history that
// lets a customer's chat survive a restart. The store's business data lives on
// the backend team's SQL Server and is not managed here.
//
// The app creates its own schema: AsyncPostgresSaver.setup() in app.py's
// lifespan() creates the checkpoint tables on first boot, so this instance
// needs no migrations or manual SQL.
//
// Usage:
//   terraform init
//   terraform apply -var="db_password=..." -var="vpc_id=..." -var="app_security_group_id=..."
//
// Never commit a real password. Pass it at apply time, or better, let
// `manage_master_user_password` hand it to Secrets Manager (see below).

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  description = "Region to deploy into. Keep this the same as the EC2 instance - a cross-region hop adds ~0.3s to every connection."
  type        = string
  default     = "eu-west-1"
}

variable "vpc_id" {
  description = "VPC of the EC2 instance running the app."
  type        = string
}

variable "private_subnet_ids" {
  description = "At least two private subnets, in different AZs (RDS requires two even for a single-AZ instance)."
  type        = list(string)
}

variable "app_security_group_id" {
  description = "Security group attached to the EC2 instance. Only this SG is allowed to reach the database."
  type        = string
}

variable "db_name" {
  type    = string
  default = "ragchat"
}

variable "db_username" {
  type    = string
  default = "raguser"
}

variable "db_password" {
  description = "Master password. Omit and set manage_master_user_password=true to let AWS generate and rotate it in Secrets Manager."
  type        = string
  default     = null
  sensitive   = true
}

variable "instance_class" {
  description = "Checkpoint writes are small and infrequent - this workload does not need a large instance."
  type        = string
  default     = "db.t4g.micro"
}

provider "aws" {
  region = var.aws_region
}

// --------------------------------------------------------------------------
// Networking: the database is private. Only the app's security group may
// reach port 5432 - referencing the SG rather than a CIDR means the rule
// keeps working when the EC2 instance is replaced and its IP changes.
// --------------------------------------------------------------------------
resource "aws_security_group" "db" {
  name        = "rag-chat-db"
  description = "Chat-persistence Postgres. Reachable only from the app."
  vpc_id      = var.vpc_id

  tags = { Name = "rag-chat-db" }
}

resource "aws_vpc_security_group_ingress_rule" "from_app" {
  security_group_id            = aws_security_group.db.id
  referenced_security_group_id = var.app_security_group_id
  from_port                    = 5432
  to_port                      = 5432
  ip_protocol                  = "tcp"
  description                  = "Postgres from the application instance only"
}

resource "aws_db_subnet_group" "this" {
  name       = "rag-chat-db"
  subnet_ids = var.private_subnet_ids

  tags = { Name = "rag-chat-db" }
}

// --------------------------------------------------------------------------
// The instance
// --------------------------------------------------------------------------
resource "aws_db_instance" "chat_persistence" {
  identifier     = "rag-chat-db"
  engine         = "postgres"
  engine_version = "16"
  instance_class = var.instance_class

  db_name  = var.db_name
  username = var.db_username

  // Either an explicit password, or AWS-managed in Secrets Manager.
  password                    = var.db_password
  manage_master_user_password = var.db_password == null ? true : null

  allocated_storage     = 20
  max_allocated_storage = 100 // autoscale rather than run out of disk
  storage_type          = "gp3"
  storage_encrypted     = true

  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible    = false // never expose this to the internet

  backup_retention_period = 7
  skip_final_snapshot     = false
  final_snapshot_identifier = "rag-chat-db-final"
  deletion_protection     = true

  auto_minor_version_upgrade = true

  tags = {
    Name    = "rag-chat-db"
    Purpose = "LangGraph chat checkpoints"
  }
}

// --------------------------------------------------------------------------
// Outputs. The endpoint is not a secret; the password is never output.
// --------------------------------------------------------------------------
output "db_endpoint" {
  description = "Host:port for CHECKPOINT_DB_HOST / CHECKPOINT_DB_PORT."
  value       = aws_db_instance.chat_persistence.endpoint
}

output "db_name" {
  value = aws_db_instance.chat_persistence.db_name
}

output "managed_password_secret_arn" {
  description = "Set only when AWS manages the master password. Read the value from Secrets Manager, not from Terraform state."
  value       = try(aws_db_instance.chat_persistence.master_user_secret[0].secret_arn, null)
}

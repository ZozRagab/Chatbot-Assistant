// Chat-persistence database for the grocery ecommerce RAG assistant.
//
// Holds ONLY LangGraph's checkpoint_* tables - the conversation history that
// lets a customer's chat survive a restart. The store's business data lives on
// the backend team's SQL Server and is not managed here.
//
// SAFETY: every resource below is NEW and named "grocery-assistant-chat-db*".
// This file creates resources; it never imports, modifies, or reads existing
// ones except by REFERENCE (your existing VPC id, your existing subnet ids,
// and your existing EC2 security group id, passed in as variables below - so
// Terraform only ever adds an ingress rule that points AT that security group,
// it never edits the group itself). Nothing else on your AWS account is
// touched. Run `terraform plan` first and confirm every line is "will be
// created" before `terraform apply`.
//
// The app creates its own schema: AsyncPostgresSaver.setup() in app.py's
// lifespan() creates the checkpoint tables on first boot, so this instance
// needs no migrations or manual SQL.
//
// Usage:
//   cd infra
//   terraform init
//   terraform plan  -var="vpc_id=vpc-xxxx" -var="private_subnet_ids=[\"subnet-aaa\",\"subnet-bbb\"]" -var="app_security_group_id=sg-xxxx"
//   terraform apply -var="vpc_id=vpc-xxxx" -var="private_subnet_ids=[\"subnet-aaa\",\"subnet-bbb\"]" -var="app_security_group_id=sg-xxxx"
//
// vpc_id, the subnet ids, and app_security_group_id are NOT secrets - they're
// just AWS resource identifiers, safe to note down or paste anywhere. Only
// db_password (if you set one instead of using the AWS-managed option below)
// is sensitive.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ------------------------------------------------------------------------
# Project naming - change this in one place if you want a different prefix.
# ------------------------------------------------------------------------
locals {
  name = "grocery-assistant-chat-db"
}

variable "aws_region" {
  description = "Region to deploy into. Keep this the same as the EC2 instance - a cross-region hop adds ~0.3s to every connection."
  type        = string
  default     = "eu-west-1"
}

variable "vpc_id" {
  description = "VPC of the EXISTING EC2 instance running the app. This RDS instance is placed in that same VPC - nothing about the VPC itself is changed."
  type        = string
}

variable "private_subnet_ids" {
  description = "At least two EXISTING private subnets, in different AZs (RDS requires two even for a single-AZ instance). Not modified - only referenced."
  type        = list(string)
}

variable "app_security_group_id" {
  description = "The EXISTING security group attached to the EC2 instance. This module does not modify that group - it only creates a NEW security group (for the database) with one ingress rule that references this id, so the database becomes reachable from the app without touching the app's own security group."
  type        = string
}

variable "db_name" {
  type    = string
  default = "grocery_assistant_chat"
}

variable "db_username" {
  type    = string
  default = "chatapp"
}

variable "db_password" {
  description = "Master password. Leave as null (default) to let AWS generate and manage it in Secrets Manager instead - recommended, since it means the password never appears in your terraform command history or .tfvars file."
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
// keeps working when the EC2 instance is replaced and its IP changes. This
// creates a brand-new security group; it does not modify var.app_security_group_id.
// --------------------------------------------------------------------------
resource "aws_security_group" "db" {
  name        = "${local.name}-sg"
  description = "Chat-persistence Postgres for the grocery assistant. Reachable only from the app instance."
  vpc_id      = var.vpc_id

  tags = {
    Name    = "${local.name}-sg"
    Project = "grocery-assistant"
    Purpose = "chat-persistence-db"
  }
}

resource "aws_vpc_security_group_ingress_rule" "from_app" {
  security_group_id            = aws_security_group.db.id
  referenced_security_group_id = var.app_security_group_id
  from_port                    = 5432
  to_port                      = 5432
  ip_protocol                  = "tcp"
  description                  = "Postgres from the grocery-assistant app instance only"
}

resource "aws_db_subnet_group" "this" {
  name       = "${local.name}-subnets"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name    = "${local.name}-subnets"
    Project = "grocery-assistant"
  }
}

// --------------------------------------------------------------------------
// The instance
// --------------------------------------------------------------------------
resource "aws_db_instance" "chat_persistence" {
  identifier     = local.name
  engine         = "postgres"
  engine_version = "16"
  instance_class = var.instance_class

  db_name  = var.db_name
  username = var.db_username

  // Either an explicit password, or AWS-managed in Secrets Manager (default).
  password                    = var.db_password
  manage_master_user_password = var.db_password == null ? true : null

  allocated_storage     = 20
  max_allocated_storage = 100 // autoscale rather than run out of disk
  storage_type          = "gp3"
  storage_encrypted     = true

  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible    = false // never expose this to the internet

  backup_retention_period   = 7
  skip_final_snapshot       = false
  final_snapshot_identifier = "${local.name}-final"
  deletion_protection       = true

  auto_minor_version_upgrade = true

  tags = {
    Name    = local.name
    Project = "grocery-assistant"
    Purpose = "LangGraph chat checkpoints"
  }
}

// --------------------------------------------------------------------------
// Outputs. The endpoint is not a secret; the password is never output.
// --------------------------------------------------------------------------
output "db_endpoint" {
  description = "Host:port for CHECKPOINT_DB_HOST / CHECKPOINT_DB_PORT, or to build CHECKPOINT_DB_URL."
  value       = aws_db_instance.chat_persistence.endpoint
}

output "db_name" {
  value = aws_db_instance.chat_persistence.db_name
}

output "db_username" {
  value = aws_db_instance.chat_persistence.username
}

output "managed_password_secret_arn" {
  description = "Set only when AWS manages the master password (the default). Retrieve the actual password from Secrets Manager - it is never written to Terraform state or output here in plaintext."
  value       = try(aws_db_instance.chat_persistence.master_user_secret[0].secret_arn, null)
}

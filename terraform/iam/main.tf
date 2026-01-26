terraform {
  required_version = ">= 1.14.0"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "~> 4.0"        # Use a version of the AWS provider that is compatible with version
    }
  }
  backend "local" {}
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}
locals {
  current_user_name = element(
    split("/", data.aws_caller_identity.current.arn),
    length(split("/", data.aws_caller_identity.current.arn)) - 1
  )
}

data "aws_region" "current" {}

output "current_user_name" {
  value = local.current_user_name
}



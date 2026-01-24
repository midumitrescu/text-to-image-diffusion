terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.28"
    }

    docker = {
      source  = "kreuzwerker/docker"
      version = "3.6.2"
    }
  }

  backend "local" {}
}

provider "aws" {
  region = "eu-central-1"
}


resource "aws_s3_bucket" "tf_state" {
  bucket = "vae-tf-state-bucket"
  force_destroy = false
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

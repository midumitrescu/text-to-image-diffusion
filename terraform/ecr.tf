provider "aws" {
  alias  = "ecr"
  region = "eu-central-1"
  assume_role {
    role_arn     = "arn:aws:iam::582821022290:role/AWSRoleECRCreation"
    session_name = "TerraformECRCreation"
  }
}

resource "aws_ecr_repository" "lambda_repo" {
  provider             = aws.ecr
  name                 = "vae-lambda"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}
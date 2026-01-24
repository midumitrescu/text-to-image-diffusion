provider "aws" {
  alias  = "push-docker"
  region = "eu-central-1"
  assume_role {
    role_arn     = "arn:aws:iam::582821022290:role/ECRPushDocker"
    session_name = "TerraformECRPushDocker"
  }
}

provider "docker" {
  host = "unix:///var/run/docker.sock"
}

data "aws_ecr_authorization_token" "auth" {
  provider = aws.push-docker
}

resource "docker_image" "vae" {

  count = var.build_docker ? 1 : 0

  name     = "${aws_ecr_repository.lambda_repo.repository_url}:${var.docker-image-version}"
  build {
    context    = "../"
    dockerfile = "Dockerfile"

    build_args = {
      VAE_VERSION = var.docker-image-version
    }
  }
}

resource "docker_registry_image" "lambda_docker_image" {

  count = var.build_docker ? 1 : 0

  name = docker_image.vae[0].name

  keep_remotely = true

  auth_config {
    address  = data.aws_ecr_authorization_token.auth.proxy_endpoint
    username = data.aws_ecr_authorization_token.auth.user_name
    password = data.aws_ecr_authorization_token.auth.password
  }
}

locals {
  #last-image_name = var.build_docker ? docker_image.vae[0].name : data.terraform_remote_state.docker.outputs.last-image_name
  last-image_name = docker_image.vae[0].name
  #image_id = var.build_docker ? docker_image.vae[0].image_id : data.terraform_remote_state.docker.outputs.image_digest
  image_id = docker_image.vae[0].image_id
}


output "last_image_name" {
  value = local.last-image_name
}


output "image_digest" {
  value = docker_image.vae[0].image_id
}
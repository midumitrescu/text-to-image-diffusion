variable "docker-image-version" {
  type = string
  description = "Set a unique docker image version to push to ECR."
  default = "0.0.3"
}

variable "build_docker" {
  type    = bool
  default = true
}
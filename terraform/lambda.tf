provider "aws" {
  alias  = "create-lambda"
  region = "eu-central-1"
  assume_role {
    role_arn     = aws_iam_role.lambda_deploy.arn
    session_name = "create-lambda"
  }
}

locals {
  image_digest = var.build_docker ? docker_image.vae[0].image_id : ""
}


resource "aws_lambda_function" "vae" {

  #depends_on = [aws_iam_policy.devs_assume_lambda_role]

  provider = aws.create-lambda
  function_name = "vae-docker-container-lambda"
  role          = aws_iam_role.lambda_execution.arn

  package_type = "Image"
  image_uri    = "582821022290.dkr.ecr.eu-central-1.amazonaws.com/vae-lambda:latest"

  timeout = 30
  memory_size = 2048
}

resource "aws_api_gateway_rest_api" "api" {
  name = "vae-api"
}

resource "aws_api_gateway_resource" "invoke" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = aws_api_gateway_rest_api.api.root_resource_id
  path_part   = "invoke"
}

resource "aws_api_gateway_method" "post" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_resource.invoke.id
  http_method   = "POST"
  authorization = "NONE"
  api_key_required = true
}

resource "aws_api_gateway_integration" "lambda" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  resource_id = aws_api_gateway_resource.invoke.id
  http_method = aws_api_gateway_method.post.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.vae.invoke_arn
}

resource "aws_api_gateway_deployment" "deploy" {
  rest_api_id = aws_api_gateway_rest_api.api.id

  depends_on = [
    aws_api_gateway_integration.lambda
  ]
}

resource "aws_api_gateway_stage" "prod" {
  deployment_id = aws_api_gateway_deployment.deploy.id
  rest_api_id   = aws_api_gateway_rest_api.api.id
  stage_name    = "prod"
}


resource "aws_api_gateway_api_key" "key" {
  name = "vae-api-key"
  enabled = true
}

resource "aws_api_gateway_usage_plan" "plan" {
  name = "vae-usage-plan"

  api_stages {
    api_id = aws_api_gateway_rest_api.api.id
    stage  = aws_api_gateway_stage.prod.stage_name
  }
}

resource "aws_api_gateway_usage_plan_key" "key_attach" {
  key_id        = aws_api_gateway_api_key.key.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.plan.id
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.vae.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.api.execution_arn}/*/*"
}

output "api_key_value" {
  value     = aws_api_gateway_api_key.key.value
  sensitive = true
}
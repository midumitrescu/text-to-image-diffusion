provider "aws" {
  alias  = "create_lambda_roles"
  region = "eu-central-1"
  assume_role {
    role_arn     = aws_iam_role.ci_agent_role.arn
    session_name = "LambdaRoles"
  }
}

resource "aws_iam_policy" "lambda_deploy_policy" {
  provider = aws.create_lambda_roles

  name        = "lambda-deploy-policy"
  description = "Allows Devs to deploy and manage Lambda functions and pass execution role"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "lambda:CreateFunction",
          "lambda:UpdateFunctionCode",
          "lambda:UpdateFunctionConfiguration",
          "lambda:DeleteFunction",
          "lambda:GetFunction",
          "lambda:ListVersionsByFunction"
        ]
        Resource = "arn:aws:lambda:${data.aws_region.current.id}:${data.aws_caller_identity.current.account_id}:function:*"
      },
      {
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = aws_iam_role.lambda_execution.arn
      }
    ]
  })
}

resource "aws_iam_role" "lambda_deploy" {
  provider = aws.create_lambda_roles
  name     = "lambda-deploy-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_deploy_attach" {
  provider = aws.create_lambda_roles

  role       = aws_iam_role.lambda_deploy.name
  policy_arn = aws_iam_policy.lambda_deploy_policy.arn
}


resource "aws_iam_group_policy" "devs_assume_role" {
  provider = aws.create_lambda_roles
  name     = "devs-assume-lambda-deployer-role"
  group    = "Devs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "sts:AssumeRole"
        Resource = aws_iam_role.lambda_deploy.arn
      }
    ]
  })
}

# TODO: Remove mihai. Do it as all other
resource "aws_iam_role" "lambda_execution" {
  provider = aws.create_lambda_roles
  name     = "vae-lambda-execution-role"

  assume_role_policy = jsonencode({
    Statement = [
      {
        Effect = "Allow"
        Principal = { Service = "lambda.amazonaws.com" }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_policy" "devs_assume_lambda_role" {
  provider = aws.create_lambda_roles

  name        = "devs-assume-lambda-execution-role"
  description = "Allow Devs group to assume the Lambda execution role"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "sts:AssumeRole"
        Resource = aws_iam_role.lambda_execution.arn
      }
    ]
  })
}

resource "aws_iam_group_policy_attachment" "attach_dev_assume_lambda" {
  provider = aws.create_lambda_roles

  group      = "Devs"
  policy_arn = aws_iam_policy.devs_assume_lambda_role.arn
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {

  provider   = aws.create_lambda_roles
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_policy" "lambda_execution_policy" {
  provider = aws.create_lambda_roles

  name        = "lambda-execution-policy"
  description = "Policy for Lambda to access logs and ECR"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_execution_attach" {
  provider   = aws.create_lambda_roles
  role       = aws_iam_role.lambda_execution.name
  policy_arn = aws_iam_policy.lambda_execution_policy.arn
}
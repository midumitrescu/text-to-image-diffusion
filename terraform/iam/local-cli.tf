resource "aws_iam_policy" "iam-deploy_policy" {

  name        = "IAM-Deloy-AWSPolicy"
  description = "Allows to create other roles and policies, used downstream for deploying the VAE AWS Lambda. Initially done by users in the cli, but easily extensible to cli agents via a special role"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "iam:CreateRole",

          "iam:CreatePolicy",
          "iam:CreatePolicyVersion",


          "iam:ListRoles",
          "iam:ListRolePolicies",
          "iam:ListRoleTags",
          "iam:ListAttachedRolePolicies",
          "iam:ListInstanceProfilesForRole",
          "iam:ListPolicyVersions",
          "iam:ListPolicyTags",
          "iam:ListAttachedGroupPolicies",
          "iam:ListAttachedUserPolicies",

          "iam:GetRole",
          "iam:GetRolePolicy",
          "iam:GetPolicy",
          "iam:GetPolicyVersion",
          "iam:GetGroupPolicy",

          "iam:AttachRolePolicy",
          "iam:AttachGroupPolicy",
          "iam:AttachUserPolicy",
          "iam:DetachRolePolicy",
          "iam:DetachGroupPolicy",
          "iam:PutGroupPolicy",
          "iam:UpdateAssumeRolePolicy",

          "iam:DeleteGroupPolicy",
          "iam:DeletePolicy",
          "iam:DeleteRole",
          "iam:DeletePolicyVersion",
          "iam:DeleteRolePolicy"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role" "dev_agent_role" {
  name = "Dev-AwsRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = [
            "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root", data.aws_caller_identity.current.arn
          ]
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "devs_role_policy_attach" {

  role       = aws_iam_role.dev_agent_role.name
  policy_arn = aws_iam_policy.iam-deploy_policy.arn
}

resource "aws_iam_role" "ci_agent_role" {
  name = "CIAgent-AwsRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = [
            "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root",
            data.aws_caller_identity.current.arn
          ]
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ci_agent_role_policy_attach" {

  role       = aws_iam_role.ci_agent_role.name
  policy_arn = aws_iam_policy.iam-deploy_policy.arn
}


resource "aws_iam_group_policy" "devs_assume_deployment_role" {
  name  = "DevsCanDeploy-AWSPolicy"
  group = aws_iam_group.developers.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "sts:AssumeRole"
        Resource = aws_iam_role.ci_agent_role.arn
      }
    ]
  })
}

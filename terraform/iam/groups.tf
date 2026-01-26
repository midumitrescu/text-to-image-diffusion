resource "aws_iam_group" "developers" {
  name = "cliDevs"
  path = "/"
}

resource "aws_iam_user_group_membership" "current_user_devs" {
  user = local.current_user_name

  groups = [
    aws_iam_group.developers.name
  ]
}
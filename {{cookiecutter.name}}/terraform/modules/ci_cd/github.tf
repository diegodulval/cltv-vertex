terraform {
  required_providers {
    github = {
      source  = "integrations/github"
      version = ">=4.0"
    }
  }
}

resource "github_repository_webhook" "github_ci_cd_trigger" {
  for_each = var.ci_cd_triggers

  active     = true
  repository = var.github_repo.name
  events     = each.value.pull_request ? ["pull_request"] : ["push"]

  configuration {
    url          = null_resource.cloudbuild_output[each.key].triggers.url
    content_type = "json"
    insecure_ssl = false
  }
}

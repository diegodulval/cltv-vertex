locals {
  cloudbuild_path    = "{{cookiecutter.name}}/terraform/${path.module}/cloudbuild.yaml"
  cloudbuild_ci_path = "{{cookiecutter.name}}/terraform/${path.module}/cloudbuild_ci.yaml"
}

resource "google_cloudbuild_trigger" "master_pr_trigger" {
  project  = var.project
  filename = local.cloudbuild_ci_path
  name     = "vertex-ai-master-pr"

  github {
    owner = var.repo_owner
    name  = var.repo_name
    pull_request {
      branch = "master"
    }
  }

  substitutions = {
    _PROJECT_ID = var.project
  }
  tags           = []
  ignored_files  = []
  included_files = []
}

resource "google_cloudbuild_trigger" "master_trigger" {
  project  = var.project
  filename = local.cloudbuild_path
  name     = "vertex-ai-master"

  github {
    owner = var.repo_owner
    name  = var.repo_name
    push {
      branch = "master"
    }
  }

  substitutions = {
    _PROJECT_ID = var.project
  }
  tags           = []
  ignored_files  = []
  included_files = []
}

resource "google_cloudbuild_trigger" "dev_pr_trigger" {
  project  = var.project
  filename = local.cloudbuild_ci_path
  name     = "vertex-ai-dev-pr"

  github {
    owner = var.repo_owner
    name  = var.repo_name
    pull_request {
      branch = "dev"
    }
  }

  substitutions = {
    _PROJECT_ID = var.project
  }
  tags           = []
  ignored_files  = []
  included_files = []
}

resource "google_cloudbuild_trigger" "dev_trigger" {
  project  = var.project
  filename = local.cloudbuild_path
  name     = "vertex-ai-dev"

  github {
    owner = var.repo_owner
    name  = var.repo_name
    push {
      branch = "dev"
    }
  }

  substitutions = {
    _PROJECT_ID = var.project,
    _TAG_NAME   = "dev"
  }
  tags           = []
  ignored_files  = []
  included_files = []
}

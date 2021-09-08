terraform {
  required_version = ">=0.13.5"

  required_providers {
    google = {
      version = ">=3.65.0"
    }
    github = {
      source  = "integrations/github"
      version = ">=4.0"
    }
  }

  backend "gcs" {}
}

provider "google" {
  project = var.project
  region  = var.region
  zone    = var.zone
}

provider "github" {
  base_url = var.github_repo.base_url == "https://github.com" ? null : var.github_repo.base_url
  owner    = var.github_repo.owner
  token    = var.github_repo.token
}

module "prerequisites" {
  source = "./modules/prerequisites"

  project = var.project
  region  = var.region
  zone    = var.zone
}

module "gcs" {
  depends_on = [module.prerequisites]
  source     = "./modules/gcs"

  project = var.project
  region  = var.region
  zone    = var.zone
  bucket  = var.bucket
}

module "cloudbuild" {
  depends_on = [module.prerequisites]
  source     = "./modules/cloudbuild"

  project    = var.project
  repo_owner = var.repo_owner
  repo_name  = var.repo_name
}

module "container_registry" {
  depends_on = [module.prerequisites]
  source     = "./modules/container_registry"

  project      = var.project
  region       = var.region
  zone         = var.zone
  gcr_location = var.gcr_location
}

module "vertex_ai" {
  depends_on = [module.prerequisites]
  source     = "./modules/vertex_ai"

  project = var.project
  region  = var.region
  zone    = var.zone
}

# module "ci_cd" {
#   # NOTE: This module needs to be the last to run, since it triggers the initial build
#   depends_on = [module.gcs]
#   source     = "./modules/ci_cd"
#
#   project        = var.project
#   region         = var.region
#   zone           = var.zone
#   github_repo    = var.github_repo
#   ci_cd_triggers = var.ci_cd_triggers
#   initial_builds = var.initial_builds
# }

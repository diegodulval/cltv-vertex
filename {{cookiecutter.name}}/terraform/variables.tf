variable "project" {
  type = string
}

variable "region" {
  type = string
}

variable "zone" {
  type = string
}

variable "bucket" {
  type = object({
    name     = string
    location = string
  })
}

variable "github_repo" {
  type = object({
    base_url = string
    owner    = string
    name     = string
    token    = string
  })
}

variable "ci_cd_triggers" {
  type = map(
    object({
      config        = string
      substitutions = map(string)
      branch        = string
      pull_request  = bool
    })
  )
}

variable "initial_builds" {
  type = map(
    object({
      config        = string
      substitutions = map(string)
      branch        = string
    })
  )
}

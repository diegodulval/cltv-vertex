locals {
  temporary_dir   = abspath("${path.module}/temp")
  url_output_file = "${local.temporary_dir}/output/url_output"

  # NOTE: Escaping below is necessary for shell execution
  substitutions_pr = {
    "_GITHUB_REPO_URL"     = "\\$(body.pull_request.head.repo.clone_url)"
    "_GITHUB_PR_URL"       = "\\$(body.pull_request.url)"
    "_GITHUB_ACTION"       = "\\$(body.action)"
    "_GITHUB_RAW_REF"      = "\\$(body.pull_request.head.ref)"
    "_GITHUB_REF"          = "\\$${_GITHUB_RAW_REF}"
    "_GITHUB_SHA"          = "\\$(body.pull_request.head.sha)"
    "_GITHUB_BASE_RAW_REF" = "\\$(body.pull_request.base.ref)"
    "_GITHUB_BASE_REF"     = "\\$${_GITHUB_BASE_RAW_REF}"
    "_GITHUB_BASE_SHA"     = "\\$(body.pull_request.base.sha)"
  }
  filter_pr = "_GITHUB_BASE_REF.matches(\\\"^%s$\\\") && _GITHUB_ACTION.matches(\\\"^opened|synchronize$\\\")"

  substitutions_push = {
    "_GITHUB_REPO_URL"     = "\\$(body.repository.clone_url)"
    "_GITHUB_PR_URL"       = ""
    "_GITHUB_ACTION"       = "push"
    "_GITHUB_RAW_REF"      = "\\$(body.ref)"
    "_GITHUB_REF"          = "\\$${_GITHUB_RAW_REF/refs\\/heads\\//}"
    "_GITHUB_SHA"          = "\\$(body.after)"
    "_GITHUB_BASE_RAW_REF" = "\\$(body.ref)"
    "_GITHUB_BASE_REF"     = "\\$${_GITHUB_BASE_RAW_REF/refs\\/heads\\//}"
    "_GITHUB_BASE_SHA"     = "\\$(body.after)"
  }
  filter_push = "_GITHUB_REF.matches(\\\"^%s$\\\")"

}

resource "null_resource" "cloudbuild_exec" {
  depends_on = [google_secret_manager_secret_version.webhook_secret_version, null_resource.api_key_output]
  for_each   = var.ci_cd_triggers

  triggers = {
    project          = var.project
    name             = "ci-cd-trigger-${each.key}"
    secret           = google_secret_manager_secret_version.webhook_secret_version.name
    api_key          = null_resource.api_key_output.triggers.api_key_name
    config_file      = "${path.module}/configs/${each.value.config}"
    config_file_hash = filesha1("${path.module}/configs/${each.value.config}")

    substitutions = join(",", [
      for k, v in merge(
        each.value.substitutions,
        each.value.pull_request ? local.substitutions_pr : local.substitutions_push
      ) : "${upper(k)}=${v}"
    ])

    filter = format(
      each.value.pull_request ? local.filter_pr : local.filter_push,
      each.value.branch
    )

    url_file = "${local.url_output_file}-${each.key}"
  }

  provisioner "local-exec" {
    command = <<-EOT
        ${path.module}/data/cloudbuild.sh create \
            ${self.triggers.project} \
            ${self.triggers.name} \
            ${self.triggers.secret} \
            "${self.triggers.api_key}" \
            ${self.triggers.config_file} \
            "${self.triggers.substitutions}" \
            "${self.triggers.filter}" \
            ${self.triggers.url_file}
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = "${path.module}/data/cloudbuild.sh delete ${self.triggers.project} ${self.triggers.name}"
  }
}

resource "null_resource" "cloudbuild_output" {
  depends_on = [null_resource.cloudbuild_exec]
  for_each   = null_resource.cloudbuild_exec

  triggers = {
    id  = each.value.id
    url = fileexists(each.value.triggers.url_file) ? chomp(file(each.value.triggers.url_file)) : null
  }

  lifecycle {
    ignore_changes = [
      triggers
    ]
  }
}

resource "null_resource" "cloudbuild_submit_initial" {
  depends_on = [null_resource.cloudbuild_output]
  for_each   = var.initial_builds

  provisioner "local-exec" {
    command = <<-EOT
        ${path.module}/data/cloudbuild.sh submit \
            ${var.project} \
            ${path.module}/configs/${each.value.config} \
            "${join(",", [for k, v in merge(each.value.substitutions, {
                "_GITHUB_REPO_URL" = "${var.github_repo.base_url}/${var.github_repo.owner}/${var.github_repo.name}"
                "_GITHUB_SHA"      = "${each.value.branch}"
            }) : "${upper(k)}=${v}"])}"
    EOT
  }
}

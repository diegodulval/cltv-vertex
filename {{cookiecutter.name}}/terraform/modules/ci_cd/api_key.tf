locals {
  api_key_file = "${local.temporary_dir}/output/api-key.json"
}

resource "google_project_service" "enable_api_keys" {
  project = var.project
  service = "apikeys.googleapis.com"

  disable_on_destroy = false
}

resource "random_string" "api_key_id" {
  length  = 8
  special = false
}

resource "null_resource" "api_key_exec" {
  depends_on = [google_project_service.enable_api_keys]

  triggers = {
    project      = var.project
    api_key_name = "Cloud Build CI/CD key - ${random_string.api_key_id.result}"
    api_key_file = local.api_key_file
  }

  provisioner "local-exec" {
    command = <<-EOT
        ${path.module}/data/api_key.sh create \
            ${self.triggers.project} \
            '${self.triggers.api_key_name}'\
            ${self.triggers.api_key_file}
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = "${path.module}/data/api_key.sh delete ${self.triggers.project} '${self.triggers.api_key_name}'"
  }
}

resource "null_resource" "api_key_output" {
  depends_on = [null_resource.api_key_exec]

  triggers = {
    id            = null_resource.api_key_exec.id
    api_key_name  = null_resource.api_key_exec.triggers.api_key_name
    api_key_id    = fileexists(local.api_key_file) ? jsondecode(file(local.api_key_file))["id"] : null
    api_key_value = fileexists(local.api_key_file) ? jsondecode(file(local.api_key_file))["value"] : null
  }

  lifecycle {
    ignore_changes = [
      triggers
    ]
  }
}

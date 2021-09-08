resource "google_project_service" "enable_secret_manager" {
  project = var.project
  service = "secretmanager.googleapis.com"

  disable_on_destroy = false
}

resource "random_string" "webhook_secret_data" {
  length  = 16
  special = false
}

resource "google_secret_manager_secret" "webhook_secret" {
  depends_on = [google_project_service.enable_secret_manager]

  project   = var.project
  secret_id = "ci-cd-webhook-secret"

  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "webhook_secret_version" {
  secret = google_secret_manager_secret.webhook_secret.id

  secret_data = random_string.webhook_secret_data.result
}

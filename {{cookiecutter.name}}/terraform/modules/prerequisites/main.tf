resource "google_project_service" "enable_service_usage" {
  project = var.project
  service = "serviceusage.googleapis.com"

  disable_on_destroy = false
}

resource "google_project_service" "enable_resource_management" {
  depends_on = [google_project_service.enable_service_usage]

  project = var.project
  service = "cloudresourcemanager.googleapis.com"

  disable_on_destroy = false
}

resource "google_project_service" "enable_service_management" {
  depends_on = [google_project_service.enable_resource_management]

  project = var.project
  service = "servicemanagement.googleapis.com"

  disable_on_destroy = false
}

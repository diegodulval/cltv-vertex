resource "google_container_registry" "container_registry" {
  project  = var.project
  location = var.gcr_location
}

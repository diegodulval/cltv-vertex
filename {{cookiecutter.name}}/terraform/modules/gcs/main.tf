resource "google_storage_bucket" "pipelines_bucket" {
  project  = var.project
  name     = var.bucket.name
  location = var.bucket.location

  force_destroy = true
}

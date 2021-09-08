resource "google_storage_bucket" "ltc_terraform_bucket" {
  project  = var.project
  name     = var.bucket.name
  location = var.bucket.location

  force_destroy = true
}

resource "google_project_service" "enable_vertex_ai" {
  project = var.project
  service = "aiplatform.googleapis.com"
}

resource "google_project_service" "enable_compute_engine" {
  project = var.project
  service = "compute.googleapis.com"
}

data "google_project" "project" {}

resource "google_project_iam_member" "cloudbuild_aiplatform_role" {
  project = var.project
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_sa_user_role" {
  project = var.project
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

resource "google_project_iam_member" "compute_storage_admin_role" {
  depends_on = [google_project_service.enable_compute_engine]

  project = var.project
  role    = "roles/storage.admin"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

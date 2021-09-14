resource "google_project_service" "enable_vertex_ai" {
  project = var.project
  service = "aiplatform.googleapis.com"
}

resource "google_project_service" "enable_compute_engine" {
  project = var.project
  service = "compute.googleapis.com"
}

data "google_project" "project" {}

resource "google_service_account" "vertex_ai_sa" {
  project      = var.project
  account_id   = "${var.deployment_name}-sa"
  display_name = "${var.deployment_name}-sa"
}

resource "google_project_iam_member" "vertex_ai_sa_access" {
  project = var.project
  role    = "roles/aiplatform.customCodeServiceAgent"
  member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
}

resource "google_project_iam_member" "vertex_ai_sa_user_role" {
  project = var.project
  role    = "roles/iam.serviceAccountAdmin"
  member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
}

resource "google_service_account_iam_member" "vertex_ai_sa_impersonation" {
  service_account_id = google_service_account.vertex_ai_sa.name
  role               = "roles/iam.serviceAccountAdmin"
  member             = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-aiplatform.iam.gserviceaccount.com"
}

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

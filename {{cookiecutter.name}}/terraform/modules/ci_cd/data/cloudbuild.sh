#!/bin/sh

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

create() {
    echo "Info: Creating Cloud Build webhook trigger: $2" 1>&2

    CMD="gcloud alpha builds triggers create webhook --project=$1 --name=$2 --secret=$3 --inline-config=$5"
    if [ -n "$6" ] ; then
        CMD="$CMD --substitutions='$6'"
    fi
    if [ -n "$7" ] ; then
        CMD="$CMD --filter='$7'"
    fi

    set +e
    RESPONSE=$(eval "$CMD" 2>&1)
    RETURN_CODE=$?
    set -e

    if [ "${RETURN_CODE}" -ne 0 ] && [ -z "$(echo "${RESPONSE}" | grep "Unknown transform function matches")" ]; then
        echo "${RESPONSE}"
        exit 1
    fi

    API_KEY="$(${ROOT_DIR}/api_key.sh get_key "$1" "$4")"
    if [ -z "${API_KEY}" ] ; then
        echo "Error: Failed to retrieve Google created API key value" 1>&2
        exit 1
    fi

    SECRET_VALUE="$(gcloud secrets versions access --project=$1 $3)"
    if [ -z "${SECRET_VALUE}" ] ; then
        echo "Error: Failed to retrieve latest secret value" 1>&2
        exit 1
    fi

    mkdir -p "$(dirname "$8")"
    echo "https://cloudbuild.googleapis.com/v1/projects/$1/triggers/$2:webhook?key=${API_KEY}&secret=${SECRET_VALUE}" > "$8"
}

delete() {
    echo "Info: Deleting Cloud Build webhook trigger: $2" 1>&2

    gcloud alpha builds triggers delete -q --project=$1 "$2"
}

submit() {
    echo "Info: Submitting Cloud Build run" 1>&2

    CMD="gcloud builds submit --project=$1 --config=$2 --no-source"
    if [ -n "$3" ] ; then
        CMD="$CMD --substitutions='$3'"
    fi

    eval "$CMD" 2>&1
}

case "$1" in
    create)
        shift
        create \
            "${1:?Missing argument: GCP project ID}" \
            "${2:?Missing argument: trigger name}" \
            "${3:?Missing argument: secret}" \
            "${4:?Missing argument: api_key}" \
            "${5:?Missing argument: Cloud Build config path}" \
            "${6}" \
            "${7}" \
            "${8:?Missing argument: output path}"
        ;;

    delete)
        shift
        delete "${1:?Missing argument: GCP project ID}" "${2:?Missing argument: trigger name}"
        ;;

    submit)
        shift
        submit \
            "${1:?Missing argument: GCP project ID}" \
            "${2:?Missing argument: Cloud Build config path}" \
            "${3}"
        ;;

    *)
        echo "Usage: $0 {create|delete|submit} [ARGS]" 1>&2
        exit 1
        ;;
esac

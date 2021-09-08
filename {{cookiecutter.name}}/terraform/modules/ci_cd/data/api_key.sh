#!/bin/sh

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

create() {
    echo "Info: Creating GCP API key: $2" 1>&2

    if [ -n "$(get_key "$1" "$2")" ] ; then
        echo "Error: API key already exists: $2" 1>&2
        exit 1
    fi

    mkdir -p "$(dirname "$3")"
    if ! gcloud alpha services api-keys create \
        --project "$1" \
        --display-name "$2" \
        --api-target=service=cloudbuild.googleapis.com \
        > output.txt 2>&1 ; then
        echo "Error: API call failed: $(cat output.txt)" 1>&2
        exit 1
    fi
    API_KEY_ID="$(cat output.txt | grep -o '"name":".*"' | sed 's/"name":"\(.*\)"/\1/')"
    API_KEY_VALUE="$(cat output.txt | grep -o '"keyString":".*"' | sed 's/"keyString":"\(.*\)"/\1/')"
    rm output.txt

    if [ -z "${API_KEY_ID}" ] || [ -z "${API_KEY_VALUE}" ]; then
        echo "Error: API key could not be parsed from API response" 1>&2
        exit 1
    fi

    echo "{\"id\": \"${API_KEY_ID}\", \"value\": \"${API_KEY_VALUE}\"}" > "$3"
}

delete() {
    echo "Info: Deleting GCP API key: $2" 1>&2

    API_KEY_NAME=$(get_id "$1" "$2")
    if [ -z "${API_KEY_NAME}" ] ; then
        echo "Error: API key could not be found: $2" 1>&2
        exit 1
    fi
    gcloud alpha services api-keys delete --project "$1" "${API_KEY_NAME}"
}

get_key() {
    echo "Info: Getting GCP API key value: $2" 1>&2

    API_KEY_NAME=$(get_id "$1" "$2")
    if [ -z "${API_KEY_NAME}" ] ; then
        echo "Error: API key could not be found: $2" 1>&2
        exit 1
    fi
    gcloud alpha services api-keys get-key-string --project "$1" "${API_KEY_NAME}" | grep -o 'keyString: .*' | sed 's/keyString: \(.*\)/\1/'
}

get_id() {
    echo "Info: Getting GCP API key ID: $2" 1>&2

    API_KEY_NAME=$(gcloud alpha services api-keys --project "$1" list 2>&1 | grep -o "[a-z0-9/\-]\+[ ]\+$2" | sed "s#\([a-z0-9/\-]\+\)[ ]\+$2#\1#")
    if [ -z "${API_KEY_NAME}" ] ; then
        echo "Error: API key could not be found: $2" 1>&2
        exit 1
    fi

    echo "${API_KEY_NAME}" | head -1
}

case "$1" in
    create)
        shift
        create \
            "${1:?Missing argument: GCP project ID}" \
            "${2:?Missing argument: api key name}" \
            "${3:?Missing argument: output path}"
        ;;

    delete)
        shift
        delete "${1:?Missing argument: GCP project ID}" "${2:?Missing argument: api key name}"
        ;;

    get_key)
        shift
        get_key "${1:?Missing argument: GCP project ID}" "${2:?Missing argument: api key name}"
        ;;

    get_id)
        shift
        get_id "${1:?Missing argument: GCP project ID}" "${2:?Missing argument: api key name}"
        ;;

    *)
        echo "Usage: $0 {create|delete|get_key|get_id} [ARGS]" 1>&2
        ;;
esac

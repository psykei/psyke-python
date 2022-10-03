#!/bin/sh

DT=${2:-${RETRY_TIME:-5m}}
MAX=${3:-${MAX_RETRIES:-3}}

for N in `seq 1 $MAX`; do
    echo "Attempt $N/$MAX: $1"
    eval $1;
    RESULT=$?
    if [[ $RESULT -eq 0 ]]; then
        exit 0
    fi
    if [[ $N -lt $MAX ]]; then
        echo "Failed attempt $N/$MAX. Waiting $DT"
        sleep $DT
    else
        echo "Failed attempt $N/$MAX."
        exit $RESULT
    fi
done

#!/bin/bash

# ============================================================
# This batch script is meant to be used to complete a workload
# that requires more time than allowed by a partition's wall
# time limit.
#
# There are 2 sections in this script that can be edited by
# users
#
#     * Both sections will start with
#       "BEGIN USER-EDITABLE SECTION" and end with
#       "END USER-EDITABLE SECTION"
#     * DO NOT EDIT ELSEWHERE
#
# 1. The SBATCH options and MAX_ITERATIONS
# 2. The job step and associated logic near the bottom
# 
# NOTE: Editing elsewhere can lead to infinite job submission 
#       loops that can adversely affect other users.
# ============================================================

# ------------------------------------------------------------
# BEGIN USER-EDITABLE SECTION
# ------------------------------------------------------------
#     * SBATCH options
#     * MAX_ITERATIONS
#           - Max number of iterations (i.e., jobs) that will
#             be submitted to avoid infinite loops
# ------------------------------------------------------------

#SBATCH -J resubmit-test    # Job name
#SBATCH -o %x-output.%j     # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                # Total number of nodes requested
#SBATCH -t 12:00:00         # Run time (hh:mm:ss)
#SBATCH -p mi2508x          # Desired partition      

MAX_ITERATIONS=4

# ------------------------------------------------------------
# END USER-EDITABLE SECTION
# ------------------------------------------------------------

CURRENT_JOB_ID="${SLURM_JOB_ID}"

# Keep track of ITERATION number and export so it is available in subsequent jobs
if [ -z "$ITERATION" ]; then
    export ITERATION=1
else
    export ITERATION=$((ITERATION + 1))
fi

# We're on the first iteration of the auto resubmission
if [ "$ITERATION" -eq 1 ]; then

    export LAST_JOB_ID="${CURRENT_JOB_ID}"
    echo "Submitting 2nd job."
    NEXT_JOB_ID=$(sbatch --parsable --dependency=afternotok:${CURRENT_JOB_ID} auto_resubmit.sh)

# We're on the 2nd through N-1 iteration
elif [[ "$ITERATION" -gt 1 && "$ITERATION" -lt "$MAX_ITERATIONS" ]]; then

    echo "ITERATION $ITERATION of $MAX_ITERATIONS."

    LAST_JOB_EXIT_STATUS=$(sacct -X -n -o State -j ${LAST_JOB_ID} | xargs)

    if [ "$LAST_JOB_EXIT_STATUS" == "TIMEOUT" ]; then

        export LAST_JOB_ID="${CURRENT_JOB_ID}"
        echo "Submitting next job."
        NEXT_JOB_ID=$(sbatch --parsable --dependency=afternotok:${CURRENT_JOB_ID} auto_resubmit.sh)

    else

        echo "Last job did not exit with TIMEOUT."
        echo "Last Job ID: ${LAST_JOB_ID}"
        echo "Last Job Exit Status: ${LAST_JOB_EXIT_STATUS}"
        echo "Exiting..."
        exit 0

    fi

# Do not submit a new job if this is the final iteration
elif [ "$ITERATION" -eq "$MAX_ITERATIONS" ]; then

    echo "ITERATION $ITERATION of $MAX_ITERATIONS."
    echo "No further jobs will be submitted."

# ITERATION should not be greater than MAX_ITERATIONS or less than 1
else

    echo "ITERATION: $ITERATION"
    echo "MAX_ITERATIONS $MAX_ITERATIONS"
    echo "Something went wrong. Exiting..."
    exit 1

fi

# ---------------------------------------------------------
# BEGIN USER-EDITABLE SECTION
# Begin job step and associated logic 
# ---------------------------------------------------------
#     * This is where you run an iteration (i.e., job) of
#       your overall workload.
#     * You will need to add custom logic based on your
#       application and its checkpoint system. 
#
#     E.g., something like
#     latest_checkpoint=$(ls checkpoint* | tail -n1)
#     python3 <script-name>.py --chk=latest_checkpoint
# ---------------------------------------------------------

# Activate the virtual environment
source myenv/bin/activate
python3 train.py

# ---------------------------------------------------------
# END USER-EDITABLE SECTION
# ---------------------------------------------------------

# If job step completes, cancel next job to break the cycle
if [ -n "${NEXT_JOB_ID:-}" ]; then
    scancel "$NEXT_JOB_ID"
fi

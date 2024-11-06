from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from flask import request, jsonify

# Specify the branch you want to monitor
BRANCH_TO_MONITOR = "master"


# CMD: ngrok ngrok http --url=sharing-loudly-mongrel.ngrok-free.app 8080
# url: https://admin:admin@sharing-loudly-mongrel.ngrok-free.app/api/v1/dags/github_branch_monitor/dagRuns
def github_webhook_listener():
    """
    This function listens for the GitHub webhook trigger.
    """
    # Parse GitHub webhook payload
    payload = request.get_json()
    ref = payload.get("ref", "")

    # Check if the push event is for the specific branch
    if ref == f"refs/heads/{BRANCH_TO_MONITOR}":
        # Trigger the DAG task here if branch matches
        # You might need to implement further logic to initiate the task in this context
        return jsonify({"status": "success", "message": f"Push detected on branch {BRANCH_TO_MONITOR}"}), 200
    else:
        return jsonify({"status": "ignored", "message": f"Push detected on a different branch: {ref}"}), 200


def run_task_on_push(**kwargs):
    """
    The task to run when a push is detected on the specified branch.
    """
    # Here you can add any task logic you want to perform
    print("Task is triggered by a push on the monitored GitHub branch.")


# Define the Airflow DAG
with DAG(
        dag_id="github_branch_monitor",
        description="Trigger a task when a push is detected on a specific GitHub branch",
        default_args={
            "owner": "airflow",
            "depends_on_past": False,
            "retries": 1,
        },
        schedule_interval=None,  # No schedule; triggered by webhook
        start_date=days_ago(1),
        catchup=False,
) as dag:
    # Python task to execute on push detection
    task = PythonOperator(
        task_id="run_task_on_push",
        python_callable=run_task_on_push,
        provide_context=True,
    )

watchmedo shell-command \
  --patterns="*.py" \
  --ignore-patterns="*/.git/*;*.swp;*.tmp;*/__pycache__/*" \
  --recursive \
  --command='curl -X POST http://localhost:8080/api/v1/dags/deploy_code_update_docker/dagRuns \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic YWlyZmxvdzplc3NhaUBhaXJmbG93" \
  -d "{\"conf\": {}}"' \
  --wait 5 \
  .

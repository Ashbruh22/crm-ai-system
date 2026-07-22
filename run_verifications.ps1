$ErrorActionPreference = "Continue"

Write-Host "2. Health and readiness checks:"
curl.exe -s http://localhost/health | python -m json.tool
curl.exe -s http://localhost/ready | python -m json.tool
Write-Host ""

Write-Host "3. Get a JWT token then make a prediction:"
$TOKEN = (curl.exe -s -X POST http://localhost/api/v1/auth/token -H "Content-Type: application/json" -d '{\"username\":\"test_rep\",\"password\":\"test_pass\",\"role\":\"sales_rep\"}' | ConvertFrom-Json).access_token
Write-Host "Token: $TOKEN"

curl.exe -s -X POST http://localhost/api/v1/predict/outcome -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{\"opportunity_id\":\"OPP_VERIFY_001\",\"sales_agent\":\"Sarah Chen\",\"product\":\"Enterprise Suite\",\"engage_date\":\"2024-01-15\"}' | python -m json.tool
Write-Host ""

Write-Host "4. Confirm Redis caching works — second call must return `"cached`": true:"
curl.exe -s -X POST http://localhost/api/v1/predict/outcome -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{\"opportunity_id\":\"OPP_VERIFY_001\",\"sales_agent\":\"Sarah Chen\",\"product\":\"Enterprise Suite\",\"engage_date\":\"2024-01-15\"}' | python -m json.tool
Write-Host ""

Write-Host "5. Auth checks:"
Write-Host "Unauthenticated — must return 401"
curl.exe -s -X POST http://localhost/api/v1/predict/outcome -H "Content-Type: application/json" -d '{\"opportunity_id\":\"OPP_TEST\",\"sales_agent\":\"Sarah Chen\",\"product\":\"Enterprise Suite\",\"engage_date\":\"2024-01-15\"}' | python -m json.tool

Write-Host "Non-admin hitting admin endpoint — must return 403"
curl.exe -s -X POST http://localhost/api/v1/admin/retrain -H "Authorization: Bearer $TOKEN" | python -m json.tool
Write-Host ""

Write-Host "6. SHAP global explanation:"
curl.exe -s http://localhost/api/v1/explain/global -H "Authorization: Bearer $TOKEN" | python -m json.tool
Write-Host ""

Write-Host "7. Prometheus metrics:"
curl.exe -s http://localhost/metrics | Select-Object -First 30
Write-Host ""

Write-Host "8. LSTM cycle prediction:"
curl.exe -s -X POST http://localhost/api/v1/predict/cycle -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{\"opportunity_id\":\"OPP_VERIFY_001\",\"sales_agent\":\"Sarah Chen\",\"product\":\"Starter Pack\",\"engage_date\":\"2024-01-15\",\"days_elapsed\":12,\"current_stage\":\"qualification\",\"num_activities\":3,\"days_since_last_activity\":2}' | python -m json.tool
Write-Host ""

Write-Host "9. Smoke tests:"
pytest tests/test_api_smoke.py -v
Write-Host ""

Write-Host "10. Confirm no deprecated patterns in main.py:"
grep -n "on_event\|--reload\|reload=True" app/main.py app/config.py docker-compose.yml Dockerfile
Write-Host "Done"

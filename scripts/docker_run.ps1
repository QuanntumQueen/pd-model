# Build and run the PD-model API in Docker (Windows PowerShell)
docker build -t pd-model .
docker rm -f pd-model 2>$null
docker run -d --name pd-model -p 8000:8000 pd-model
Write-Host "API is running: http://localhost:8000/docs"

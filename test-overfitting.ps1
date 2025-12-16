$ALB_URL = "http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"

# Wait for endpoint
Write-Host "Waiting for endpoint..."
do {
    Start-Sleep -Seconds 20
    $ep = Invoke-RestMethod -Uri "$ALB_URL/model-init/endpoint-status"
    Write-Host "Status: $($ep.status)"
} while ($ep.status -ne "InService")

# Test
$tests = @(
    "nice size, very clear but randomly shuts off",
    "This tablet is on the smaller side but works for me",
    "Too difficult to setup"
)

foreach ($t in $tests) {
    $body = @{ text = $t } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "$ALB_URL/predict" -Method POST -Body $body -ContentType "application/json"
    Write-Host "$t -> $($r.label) ($([math]::Round($r.confidence, 2)))"
}
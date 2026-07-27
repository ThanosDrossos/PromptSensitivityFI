# PromptSensitivityFI demo launcher (Windows, PowerShell 5.1 compatible, ASCII only).
#   powershell -ExecutionPolicy Bypass -File app\launch.ps1
# Starts Streamlit, opens the local browser, starts an ngrok tunnel with
# basic auth (kit / promptsensitivity), prints + copies the public URL.
# Ctrl+C stops both.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$user = "kit"
$pass = "promptsensitivity"
$port = 8501

Write-Host ">> starting Streamlit on http://localhost:$port ..."
$py = Join-Path $repo ".venv\Scripts\python.exe"
$st = Start-Process -PassThru -WindowStyle Hidden $py -ArgumentList @(
    "-m","streamlit","run","app/streamlit_app.py",
    "--server.headless","true","--server.port","$port")

$deadline = (Get-Date).AddSeconds(60)
$up = $false
while ((Get-Date) -lt $deadline -and -not $up) {
    try {
        $c = New-Object Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $port); $c.Close(); $up = $true
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $up) { Write-Warning "Streamlit did not open port $port in 60s"; }
Start-Process "http://localhost:$port"

Write-Host ">> starting ngrok tunnel (basic auth: $user / $pass) ..."
$ngrokCmd = Get-Command ngrok -ErrorAction SilentlyContinue
if ($ngrokCmd) {
    $ngrokExe = $ngrokCmd.Source
} else {
    $ngrokExe = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe\ngrok.exe"
}
$ng = $null
if (Test-Path $ngrokExe) {
    $ng = Start-Process -PassThru -WindowStyle Hidden $ngrokExe -ArgumentList @(
        "http","$port","--basic-auth","${user}:${pass}")
    $url = $null
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline -and -not $url) {
        Start-Sleep -Milliseconds 800
        try {
            $tunnels = (Invoke-RestMethod "http://127.0.0.1:4040/api/tunnels").tunnels
            foreach ($t in $tunnels) { if ($t.proto -eq "https") { $url = $t.public_url; break } }
        } catch { }
    }
    if ($url) {
        try { Set-Clipboard -Value $url } catch { }
        Write-Host ""
        Write-Host "==============================================================" -ForegroundColor Green
        Write-Host ("  Public URL (copied to clipboard):  " + $url) -ForegroundColor Green
        Write-Host ("  Login: " + $user + " / " + $pass) -ForegroundColor Green
        Write-Host "==============================================================" -ForegroundColor Green
    } else {
        Write-Warning "ngrok tunnel did not come up (run 'ngrok config check')."
    }
} else {
    Write-Warning "ngrok not found. Install: winget install ngrok.ngrok, then add an authtoken."
    Write-Host "Streamlit keeps running locally."
}

Write-Host ("Local app: http://localhost:" + $port + "  (Ctrl+C stops app + tunnel)")
try {
    Wait-Process -Id $st.Id
} finally {
    foreach ($proc in @($st, $ng)) {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

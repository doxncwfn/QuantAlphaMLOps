param(
    [string]$Target = "main"
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

try {
    Write-Host "[1/4] pdflatex (pass 1)..." -ForegroundColor Cyan
    & pdflatex -interaction=nonstopmode "$Target.tex" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 1 failed (exit $LASTEXITCODE). Check $Target.log." }

    Write-Host "[2/4] bibtex..." -ForegroundColor Cyan
    & bibtex $Target | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "bibtex failed (exit $LASTEXITCODE). Check $Target.blg." }

    Write-Host "[3/4] pdflatex (pass 2)..." -ForegroundColor Cyan
    & pdflatex -interaction=nonstopmode "$Target.tex" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 2 failed (exit $LASTEXITCODE). Check $Target.log." }

    Write-Host "[4/4] pdflatex (pass 3)..." -ForegroundColor Cyan
    & pdflatex -interaction=nonstopmode "$Target.tex" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 3 failed (exit $LASTEXITCODE). Check $Target.log." }

    $pdf = Get-Item "$Target.pdf"
    Write-Host "Build succeeded: $($pdf.FullName) ($([math]::Round($pdf.Length / 1KB, 1)) KB)" -ForegroundColor Green
}
finally {
    Pop-Location
}

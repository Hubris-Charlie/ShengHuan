# 一键入口。评委/组员只需要记住这一个命令。
param(
    [ValidateSet('test', 'eval', 'serve', 'all')]
    [string]$Mode = 'all',
    [switch]$Fast
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Step($name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }

switch ($Mode) {
    'test' {
        Step '单元测试 + 端到端通路测试'
        python -m unittest discover -s tests
        exit $LASTEXITCODE
    }
    'eval' {
        Step '评测集'
        if ($Fast) { python eval/run.py --fast } else { python eval/run.py }
        exit $LASTEXITCODE
    }
    'serve' {
        Step '启动网页通话台'
        Write-Host '浏览器打开 http://127.0.0.1:8000/?ws=8765' -ForegroundColor Green
        python server/ws_server.py
        exit $LASTEXITCODE
    }
    'all' {
        Step '单元测试'
        python -m unittest discover -s tests
        if ($LASTEXITCODE -ne 0) { Write-Host '测试未通过，中止' -ForegroundColor Red; exit 1 }

        Step '评测集（快速模式）'
        python eval/run.py --fast
        if ($LASTEXITCODE -ne 0) { Write-Host '评测未全通过，请查看上面失败用例' -ForegroundColor Yellow }

        Step '启动网页通话台'
        Write-Host '浏览器打开 http://127.0.0.1:8000/?ws=8765' -ForegroundColor Green
        python server/ws_server.py
    }
}

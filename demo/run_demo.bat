@echo off
rem 一键启动逃跑反射演示（纯标准库，不需要装任何包）
cd /d %~dp0..
D:\Code\FlyBrain\env\python.exe demo\server.py
pause

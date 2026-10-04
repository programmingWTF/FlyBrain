# 部署提示词（复制给服务器上的 Agent）

把下面整段复制过去。它假设 Agent 能 SSH 到一台 Linux 服务器并有 sudo。

---

```
帮我在本机部署一个网页服务。仓库是 https://github.com/programmingWTF/FlyBrain

## 这是什么

用真实果蝇全脑连接组（FlyWire FAFB v783，144,837 神经元 / 1500 万突触）
当作一个**冻结的、零可学参数**的神经网络，驱动一个 Flappy 小游戏。
后端是单进程 Python，用标准库 http.server 提供服务；前端零构建（原生 ES module）。

## 请按顺序做，每步都验证

### 1. 环境检查
    确认：python3 --version（**必须 >= 3.10**，代码用了 `X | None` 这种注解语法）
    确认：磁盘至少有 1 GB 空余（仓库约 52 MB + 虚拟环境 + pip 缓存）
    确认：8620 端口没被占用（ss -tlnp | grep 8620）
    确认：有 git（git --version）

### 2. 拉代码
    git clone https://github.com/programmingWTF/FlyBrain.git /opt/flyflappy
    cd /opt/flyflappy

    然后**必须核对 data/ 里这三个文件都在、且大小非零**：
      data/spiking_full.npz   约 50 MB    (冻结脉冲脑)
      data/spiking_full.json  约 226 KB   (元数据)
      data/coords.npz         约 1.6 MB   (解剖坐标 + 关键群，预烤)
    三个都缺一不可。用 ls -la data/ 检查。

### 3. 装依赖（CPU 版 torch 就够，不需要 GPU）
    python3 -m venv .venv
    . .venv/bin/activate
    pip install numpy pandas torch --index-url https://download.pytorch.org/whl/cpu

### 4. 先手工跑一次，确认能起来
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python demo/server.py --asset spiking_full --no-browser --port 8620

    ⚠️ 注意 --asset 必须显式写 spiking_full。
       它的默认值是 spiking_circuit，那个子图不含 LC4/LPLC2，
       而 Flappy 的所有感觉输入都打在 LC4/LPLC2 上 —— 用默认值的后果是
       脑完全不发放、**分数恒为 0**，而且**不报错**，很难查。

    起来后另开一个 shell 验证：
      curl -s http://127.0.0.1:8620/api/info | head -c 500
    返回的 JSON 必须满足：
      asset == "spiking_full"
      n_neurons == 144837
      lit 有 556 项          ← 这是坐标就位的标志；是空的说明 coords.npz 没读到

      curl -s -o /dev/null -w "%{http_code} %{size_download}\n" \
           http://127.0.0.1:8620/vendor/three.module.js
      # 期望 "200 1304820"

      curl -s http://127.0.0.1:8620/api/coords.bin | wc -c
      # 期望 1738044

      # 脑真的在发放（这一步最能说明问题）
      curl -s -X POST http://127.0.0.1:8620/api/step \
        -H 'Content-Type: application/json' \
        -d '{"ticks":4,"need_spikes":1,"drives":[{"type":"bidi","y":300,"vy":0,"gap":342,"gap_margin":18,"vy_gate":1,"ceil_boost":1,"dors_scale":0.35,"vent_gain":2.0,"vent_dev":120}],"bidi":{"s50size":30,"n":3,"gain":1,"groups":["LC4","LPLC2"]}}'
      # 返回里应该有 lc4_rate / dn01_recent / flap 等字段，且没有 error

### 5. 做成 systemd 常驻服务

    写 /etc/systemd/system/flyflappy.service：

    [Unit]
    Description=FlyBrain Flappy (frozen Drosophila connectome demo)
    After=network.target

    [Service]
    Type=simple
    User=<改成实际用户名>
    WorkingDirectory=/opt/flyflappy
    Environment=PYTHONUNBUFFERED=1
    Environment=OMP_NUM_THREADS=1
    Environment=MKL_NUM_THREADS=1
    ExecStart=/opt/flyflappy/.venv/bin/python demo/server.py --asset spiking_full --no-browser --host 127.0.0.1 --port 8620
    Restart=on-failure
    RestartSec=5

    [Install]
    WantedBy=multi-user.target

    然后：
      systemctl daemon-reload
      systemctl enable --now flyflappy
      systemctl status flyflappy --no-pager
      journalctl -u flyflappy -n 30 --no-pager

### 6. 告诉我怎么访问

    默认只监听 127.0.0.1。**不要**为了图省事直接改成 0.0.0.0 ——
    这个服务没有任何鉴权，而且每个请求都会推进 14.5 万神经元的仿真。

    优先给我 SSH 端口转发的命令，例如：
      ssh -L 8620:127.0.0.1:8620 <你的用户>@<这台机器>

    如果你判断必须对外暴露，先告诉我，我们再加 nginx 反向代理 + basic auth。

## 已知情况（不用当 bug 查）

1. **启动约 5 秒**、**内存约 480 MB**、`/api/step` 中位约 27 ms。
   本机（Ultra 9 285H）实测值，你的机器可能不同但量级一致。
2. **页面会有一点卡顿**，这是已知问题：浏览器每秒只能发 14~20 次 `/api/step`，
   而 1:1 实时需要 50 次。**是浏览器的瓶颈，不是服务器**。
   服务端单线程能跑约 37 req/s、并发能到 377 req/s。
3. **只有一颗脑（单例）**。多个人同时开页面会共享同一个脑、互相干扰。
   如果只是你自己看，忽略这条。
4. 服务启动后会**一直占着约 480 MB 内存**，这是正常的（连接组常驻）。

## 请回报给我

1. `systemctl status` 的结果
2. 第 4 步里四条 curl 的**实际输出**（特别是 /api/info 里的 asset / n_neurons / len(lit)）
3. 你用的访问方式（SSH 转发命令，或你另外配的代理）
4. 有没有哪一步和上面描述的不一样

如果哪一步失败，把**完整报错**贴给我，不要自己改代码绕过 ——
这个仓库的启动路径上有几处"缺文件不报错、只是静默退化"的地方
（比如缺 coords.npz 时 3D 脑图会空、用错 asset 时分数恒 0），
自己绕过的话会很难查。
```

---

## 备注（给你自己看的，别复制进去）

- 提示词里特意强调了两个坑，都是实践中真踩到的：
  1. **`--asset` 默认值不含 LC4/LPLC2** → 分数恒 0 且不报错
  2. **缺 `coords.npz` 只是静默退化**（3D 脑图空），不会崩

- 如果服务器上已经有 Python 环境，`pip install` 那步可以省，
  但要确认 `torch` 装的是 **CPU 版**（GPU 版在无显卡机器上装不上，或者装了也白占 2 GB）

- 如果服务器**访问不了 GitHub**（国内常见），先把仓库打包传上去：
  ```bash
  # 本地
  git bundle create flybrain.bundle --all
  scp flybrain.bundle user@server:/tmp/
  # 服务器
  git clone /tmp/flybrain.bundle /opt/flyflappy
  ```
  注意 `git bundle` 里**不含** `data/` 下那三个大文件吗？——**含**。
  它们是正常提交进仓库的，bundle 会一起带上（约 52 MB）。

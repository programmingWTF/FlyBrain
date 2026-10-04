# 部署指南

把 Flappy 那一页跑起来。**零第三方前端依赖**（three.js 随仓库自带），
后端只要 `numpy` + `torch` + `pandas`。

**本指南按 Linux 写**（`systemd` + POSIX 命令），代码本身已验证与平台无关，
见 §9。

---

## 0. 这个仓库是自包含的（重要）

运行时**只需要**下面这几个文件。核对一遍，缺哪个都会起不来：

| 文件 | 大小 | 作用 | 在仓库里？ |
|---|---|---|---|
| `data/spiking_full.npz` | 50 MB | 冻结脉冲脑（CSR 连接组：indptr/indices/codes/node_ids） | ✅ 已入库 |
| `data/spiking_full.json` | 226 KB | 元数据：`n_neurons` / `key_neurons` / `dynamics` | ✅ 已入库 |
| `data/coords.npz` | 1.6 MB | 解剖坐标 + 关键群 + 逐侧索引（**预烤**） | ✅ 已入库 |
| `demo/vendor/three.module.js` | 1.3 MB | 前端 ES module（`import 'three'`） | ✅ 已入库 |

**不需要**（历史上曾经需要，已用 `scripts/bake_coords.py` 预烤替代）：

- ❌ `data/brain/manifest.json`（51.5 MB）—— 那种"启动时读大 JSON"的慢路径已不再是必需
- ❌ `data/fafb_783_meta.feather`（13.5 MB 细胞类型表）—— 关键群与逐侧索引已烤进 `coords.npz`
- ❌ `viewer/`（全脑渲染器，那是另一个项目）

> 如果这几个文件里缺了 `coords.npz`，服务器**仍然能起**，只是右侧 3D 脑图会空
> （所有点堆在原点）；Flappy 游戏本身不受影响。启动时会打印明确提示。

---

## 1. 依赖

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy pandas torch --index-url https://download.pytorch.org/whl/cpu
```

- **需要 Python ≥ 3.10** —— 代码里用了 `pathlib.Path | None` 这种 PEP 604 注解
  （`from __future__ import annotations` 已开，但 `|` 语法本身仍要 3.10+）
- **CPU 版 torch 就够**（本 demo 是单线程 CPU 推理，1470 万突触的稀疏传播）
- 版本要求很宽：`numpy>=1.24`、`pandas>=2.0`、`torch>=2.0`
- 不需要 GPU、不需要 CUDA、不需要联网（前端资源都在仓库里）

---

## 2. 跑起来

```bash
cd flyflappy                      # 仓库根（本仓库就是 flyflappy 的内容）
python demo/server.py --asset spiking_full --no-browser --port 8620
```

然后浏览器打开 `http://<服务器地址>:8620/`。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--asset` | `spiking_circuit` | **要显式传 `spiking_full`** —— 见下面的坑 |
| `--port` | `8620` | |
| `--host` | `127.0.0.1` | **只监听本机**。要外部访问需要改成 `0.0.0.0`（见 §4） |
| `--device` | `cpu` | |
| `--no-browser` | — | 服务器上一定要加，否则它会去调 `xdg-open` |

> ⚠️ **坑：默认资产是错的。**
> `--asset` 的默认值是 `spiking_circuit`，但那个 8000 神经元的子图**不含 LC4/LPLC2**，
> 而 Flappy 的所有感觉输入都打在 LC4/LPLC2 上 —— 用默认值的结果是脑完全不发放、
> **分数恒为 0**，而且不报错。必须显式 `--asset spiking_full`。

### 启动成功的标志

正常启动会打印资产规模，并监听端口。自检：

```bash
curl -s http://127.0.0.1:8620/api/info | head -c 400
```

返回的 JSON 里应该看到：

```json
{"asset": "spiking_full", "n_neurons": 144837, "n_synapses": 15023799,
 "lit": [ ... 556 项 ... ],
 "bidi": {"groups": ["LC4","LPLC2"], "dors_scale": 0.35, "vent_gain": 2.0, ...}}
```

**`lit` 有 556 项** = 坐标就位、脑图能点亮（`LC4` 104 + `LPLC2` 210 + `LC10a` 234 + 其余）。
如果 `lit` 是空的 → `coords.npz` 没读到，检查文件在不在。

还有两个端点可以单独验证：

```bash
curl -sI http://127.0.0.1:8620/vendor/three.module.js    # 200，约 1.3 MB
curl -s  http://127.0.0.1:8620/api/coords.bin | wc -c     # 1738044 = 144837×3×4
```

---

## 3. 常驻运行

### systemd（推荐）

`/etc/systemd/system/flyflappy.service`：

```ini
[Unit]
Description=FlyBrain Flappy (frozen Drosophila connectome demo)
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/opt/flyflappy
Environment=PYTHONUNBUFFERED=1
# 单线程：这个负载是稀疏传播，多开线程反而更慢、还抢 CPU
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
ExecStart=/opt/flyflappy/.venv/bin/python demo/server.py \
          --asset spiking_full --no-browser --host 127.0.0.1 --port 8620
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now flyflappy
sudo systemctl status flyflappy
journalctl -u flyflappy -f          # 看日志
```

### 临时前台跑（调试用）

```bash
nohup python demo/server.py --asset spiking_full --no-browser --port 8620 \
      > flappy.log 2>&1 &
tail -f flappy.log
```

### 开机自启 + 只给内网

默认 `--host 127.0.0.1` 只监听本机。要在**同一台机器**上访问，
用 SSH 端口转发最安全（不用改监听地址、不暴露端口）：

```bash
ssh -L 8620:127.0.0.1:8620 your-user@your-server
# 然后本地浏览器打开 http://127.0.0.1:8620/
```

---

## 4. 对外暴露（谨慎）

⚠️ **这个服务没有任何鉴权**，而且每个请求都会推进 14.5 万神经元的仿真。
**别直接挂到公网。**

如果确实要给别人看，用反向代理加一层访问控制。nginx：

```nginx
location /flappy/ {
    auth_basic           "flyflappy";
    auth_basic_user_file /etc/nginx/.htpasswd;

    proxy_pass         http://127.0.0.1:8620/;
    proxy_http_version 1.1;
    proxy_set_header   Host $host;
    proxy_set_header   X-Real-IP $remote_addr;
    proxy_read_timeout 300s;          # /api/step（离线式调用）偶尔会慢
}

**流式注意**：`/api/game/state` 是**短轮询**（前端每帧取一次，约 400 字节），
不需要 WebSocket 或 SSE。如果反代开了响应缓冲，务必对 `/api/` 关掉
（`proxy_buffering off`），否则状态会攒着一起发，画面会一顿一顿。
```

两个必须注意的：

1. **`--host 0.0.0.0` 时没有访问控制**，只在完全可信的内网这么干
2. **一个脑 = 一份单线程状态**。多个人同时开页面会**共享同一个脑、同一个游戏**、
   互相干扰（`Session` 是单例，游戏世界也是它持有的）。
   要真支持多人，得一人一个进程 + 端口，或者改造成按会话隔离

---

## 5. 性能（本机实测）

| 指标 | 实测 | 说明 |
|---|---|---|
| **启动时间** | **4.7 s** | 读 50 MB npz + 建 1470 万条边的索引。**只发生一次** |
| **内存占用** | 约 **500 MB** 工作集 | torch 张量 + torch 自身 |
| **仿真速率** | **恒定 50.0 步/秒** | 服务端锁步线程，20ms/tick 走墙钟；实测偏差 +0.1% |
| **CPU 占用** | **持续约 1 个核** | 50 Hz 一直在跑脑 + 物理（闲置时也跑，页面随时打开都是活的） |
| **`GET /api/game/state`** | 约 **400 字节** / 次 | 前端每帧取（渲染用） |
| **`/api/step` 延迟** | 中位 **27.0 ms**（6.6 ~ 34.7） | 一次 4 个 tick；**页面已不再用它** |

> 启动 4.7 s 这个数**依赖 `data/coords.npz`**。没有它而退回读 51.5 MB 的
> `manifest.json` 时，光解析那个 JSON 就要几十秒 —— 预烤坐标就是为了消掉这一段。

### 关于那个"持续占一个核"

这是**刻意**的，不是泄漏：仿真线程按固定 50 Hz 自己往前走，
不等任何请求。这样"打开页面立刻就是活的"，也彻底摆脱了
"游戏速度取决于浏览器多快"这个结构性问题（详见 README 的
"页面 vs 评测台：差距已消除"）。

如果部署环境对 CPU 敏感，有三条路（都需要改代码，这里只记录方向）：

1. 改成**按需驱动**：没人连页面时暂停线程，有 `/api/game/state` 请求时再恢复
   （代价：切回页面时游戏是"停着的"，需要先补一段）。
2. 把 `SIM_HZ` 降到 25（游戏变 0.5 倍速，但仍然是恒定的）。
3. 用 `--device cuda`（如果机器有 GPU）—— 但要注意这个负载是稀疏传播，
   GPU 不一定更快，得实测。

---

## 6. 排障

| 现象 | 原因 | 解法 |
|---|---|---|
| `FileNotFoundError: data/spiking_full.npz` | 资产没入库/没下全 | 确认 `data/` 里有 3 个大小非零的文件 |
| **分数恒 0、脑不发放** | **用了默认资产 `spiking_circuit`**（不含 LC4/LPLC2） | 加 `--asset spiking_full` |
| 右侧 3D 面板**空白但游戏照跑** | `coords.npz` 缺失或损坏 | 日志会有 `[提示] 没找到 ...` 字样；重新拉一次仓库 |
| 3D 面板**不再闪烁/点不亮** | `/api/game/state` 没回 `viz_spike`/`lit_spike` | 这两个量必须由状态端点带上（前端自己那次 `/api/step` 已经没了）；照 README 的"两个坑"检查 |
| 游戏**不前进**（画面定住） | 仿真线程挂了 | 看日志有没有异常；`curl /api/game/state` 看 `ticks` 是否在涨 |
| 游戏**速度不对**（明显快/慢） | 主机负载过高导致线程被抢 | 正常是 50.0 步/秒；连续采样 `ticks` 的增量确认 |
| 页面白屏、控制台 `Failed to resolve module specifier "three"` | `/vendor/three.module.js` 404 | 确认 `demo/vendor/three.module.js` 存在（1.3 MB） |
| 首页 200 但所有 `/api/*` 404 | 端口被别的进程占了 | `ss -tlnp \| grep 8620`，杀掉或换端口 |
| `Address already in use` | 已在跑 | `systemctl restart flyflappy` 或换端口 |
| CPU 100% 一直不降 | 可能是**离线评测台**在跑（`scripts/flappy_bench.py`） | demo 自己约占 1 个核；再多出来就查 `ps aux \| grep flappy_bench` |

---

## 7. 重建资产（一般不需要）

`data/` 里那三个文件已经入库，正常部署**不用重建**。只在你想改数据时才需要：

```bash
# ① 重建脉冲脑资产（需要 316 MB 的原始边表，见仓库外的下载脚本）
python scripts/build_spiking_asset.py --out spiking_full --min-count 1

# ② 重烤坐标（需要 51.5 MB 的 manifest.json，来自全脑渲染项目）
python scripts/bake_coords.py \
    --manifest /path/to/data/brain/manifest.json \
    --asset spiking_full \
    --out data/coords.npz
```

① 的输入是 `data/fafb_783_simple_edgelist.feather`（302 MB）等文件，
**不在本仓库**（可重新下载）。② 的产物已经入库，所以这一步通常可以跳过。

---

## 8. 验收清单

部署完照这个顺序核一遍：

```bash
# 1) 进程在跑
systemctl is-active flyflappy        # 或 ps aux | grep server.py

# 2) 端口在听
ss -tlnp | grep 8620

# 3) 资产正确（关键：asset 必须是 spiking_full、lit 必须是 556）
curl -s http://127.0.0.1:8620/api/info \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
      print(d['asset'], d['n_neurons'], len(d['lit'])); \
      assert d['asset']=='spiking_full' and len(d['lit'])==556, '资产不对！'"

# 4) 前端资源
curl -s -o /dev/null -w "%{http_code} %{size_download}\n" \
     http://127.0.0.1:8620/vendor/three.module.js     # 期望 200 1304820

# 5) 点云
curl -s http://127.0.0.1:8620/api/coords.bin | wc -c  # 期望 1738044

# 6) 仿真在跑、而且速率恒定（**这是这次改动后最关键的验收**）
#    目标：恒定 50.0 步/秒。隔 20 秒取两次，看 ticks 的增量。
python3 - <<'PY'
import json, time, urllib.request
def st():
    return json.loads(urllib.request.urlopen(
        "http://127.0.0.1:8620/api/game/state", timeout=10).read())
a = st(); time.sleep(20); b = st()
rate = (b["ticks"] - a["ticks"]) / 20
print(f"  仿真 {rate:.2f} 步/秒（目标 50）  score {a['score']}->{b['score']}  dead={b['dead']}")
assert 45 <= rate <= 55, f"速率不对：{rate:.2f} 步/秒（应为 50 左右）"
assert b["ticks"] > a["ticks"], "仿真线程没在推进！"
print("  ✅ 恒定速率正常")
PY

# 7) 脑真的在发放（离线式调用，页面已经不用它了；这里只是确认脑活的）
#    注意：请求体不能是空对象，必须给 ticks / need_spikes / drives
curl -s -X POST http://127.0.0.1:8620/api/step \
  -H 'Content-Type: application/json' \
  -d '{"ticks":4,"need_spikes":1,
       "drives":[{"type":"bidi","y":300,"vy":0,"gap":342,
                  "gap_margin":18,"vy_gate":1,"ceil_boost":1,
                  "dors_scale":0.35,"vent_gain":2.0,"vent_dev":120}],
       "bidi":{"s50size":30,"n":3,"gain":1,"groups":["LC4","LPLC2"]}}'
# 返回里应该有 lc4_rate / dn01_recent / flap / lit_spike 等字段，且没有 error
```

最后浏览器打开，确认四件事：

1. **状态栏**显示"冻结脑 144,837 神经元 / 15.02M 突触"
2. **右侧 3D 脑图**有彩色点（LC4 橙、LPLC2 紫、DNp01 红）在闪
   —— 如果**空白或全灰**，看排障表里"3D 面板不再闪烁"那一行
3. **鸟会自己飞**（不是一直掉）—— 分数会慢慢涨
4. **速度是稳的**：鸟和管子的移动不忽快忽慢（这正是搬到服务端要解决的问题）

---

## 9. Linux 适用性（已逐项验证）

这份指南按 Linux 写。代码里的平台相关问题都查过了：

| 检查项 | 结果 |
|---|---|
| 可执行代码里的盘符路径（`D:\...`） | ✅ **没有**。项目里那些 `D:/Code/FlyBrain/env/python.exe` 全在 **docstring / 注释**里，只是示例，不影响运行 |
| 路径拼接 | ✅ 全部用 `pathlib`，**没有** `os.path.join` / 反斜杠字符串 |
| 顶层 import | ✅ 全是跨平台的：`os` / `argparse` / `json` / `pathlib` / `http.server` / `numpy` / `torch` |
| 二进制资产 | ✅ `.gitattributes` 里显式标为 `binary`，不会被行尾转换破坏（这点在 Windows 上 `core.autocrlf=true` 时尤其重要，否则 Linux clone 下来才会炸） |
| **打开浏览器** | ✅ **已修**。原来是 `subprocess.Popen(["cmd","/c","start","",url])` —— **Windows 专有**，在 Linux 上会抛 `FileNotFoundError`（被吞掉，表现为"浏览器静默不打开"）。现在改用标准库 `webbrowser.open()`，跨平台 |
| torch 线程数 | 代码不设，靠环境变量。**Linux 上一定要 `OMP_NUM_THREADS=1`**：这个负载是稀疏传播，多开会更慢还抢 CPU |

### Linux 上用不到的东西

| 文件 | 说明 |
|---|---|
| `demo/*.bat`（`run_demo.bat` / `stop_demo.bat` / `check_all.bat`） | Windows 批处理，**Linux 忽略即可**，用本指南的命令代替 |
| `demo/verify_*.js`、`demo/check_syntax.mjs` | 离线校验器，需要 Node.js。**部署不需要**，只在改代码时用 |

### 服务器上第一次跑的自检顺序

```bash
python3 -V                                  # 必须 >= 3.10
ls -la data/                                # 三个资产都在、大小非零
.venv/bin/python -c "import numpy,pandas,torch; print('ok')"
.venv/bin/python demo/server.py --asset spiking_full --no-browser --port 8620
```

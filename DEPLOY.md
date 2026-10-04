# 部署指南

把 Flappy 那一页跑起来。**零第三方前端依赖**（three.js 随仓库自带），
后端只要 `numpy` + `torch` + `pandas`。

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

- **CPU 版 torch 就够**（本 demo 是单线程 CPU 推理，147 万突触的稀疏传播）
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
    proxy_read_timeout 300s;          # /api/step 偶尔会慢
}
```

两个必须注意的：

1. **`--host 0.0.0.0` 时没有访问控制**，只在完全可信的内网这么干
2. **一个脑 = 一份单线程状态**。多个人同时开页面会**共享同一个脑**、互相干扰
   （连接是 `/api/step` 无状态调用，但 `Session` 是单例）。
   要真支持多人，得一人一个进程 + 端口，或者改造成按会话隔离

---

## 5. 性能（本机实测）

| 指标 | 实测 | 说明 |
|---|---|---|
| **启动时间** | **4.7 s** | 读 50 MB npz + 建 1470 万条边的索引。**只发生一次** |
| **内存占用** | **483 MB** 工作集 | torch 张量 + torch 自身 |
| **`/api/step` 延迟** | 中位 **27.0 ms**（6.6 ~ 34.7） | 单线程 CPU，一次 4 个 tick |
| → 单线程吞吐 | 约 **37 req/s** | 而"1:1 实时"需要 50 req/s |
| 脑/浏览器速度 | 浏览器是瓶颈 | 页面实测 14~20 tick/s，服务端并发能到 377 req/s |

> 启动 4.7 s 这个数**依赖 `data/coords.npz`**。没有它而退回读 51.5 MB 的
> `manifest.json` 时，光解析那个 JSON 就要几十秒 —— 预烤坐标就是为了消掉这一段。

**已知的卡顿来源**：浏览器的 `/api/step` 往返速度跟不上，脑会落后于物理。
这不是服务器的问题。详见 README 的"页面 vs 评测台：已定位的差距"。

---

## 6. 排障

| 现象 | 原因 | 解法 |
|---|---|---|
| `FileNotFoundError: data/spiking_full.npz` | 资产没入库/没下全 | 确认 `data/` 里有 3 个大小非零的文件 |
| **分数恒 0、脑不发放** | **用了默认资产 `spiking_circuit`**（不含 LC4/LPLC2） | 加 `--asset spiking_full` |
| 右侧 3D 面板空白 | `coords.npz` 缺失或损坏 | 日志会有 `[提示] 没找到 ...` 字样；重新拉一次仓库 |
| 页面白屏、控制台 `Failed to resolve module specifier "three"` | `/vendor/three.module.js` 404 | 确认 `demo/vendor/three.module.js` 存在（1.3 MB） |
| 首页 200 但所有 `/api/*` 404 | 端口被别的进程占了 | `ss -tlnp \| grep 8620`，杀掉或换端口 |
| `Address already in use` | 已在跑 | `systemctl restart flyflappy` 或换端口 |
| CPU 100% 一直不降 | 有人在跑 **离线评测台**（`scripts/flappy_bench.py`），不是 demo | demo 空闲时几乎不耗 CPU；查 `ps aux \| grep flappy_bench` |

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

# 6) 脑真的在发放（这一步最能说明问题）
curl -s -X POST http://127.0.0.1:8620/api/step \
     -H 'Content-Type: application/json' -d '{}' | head -c 300
# 返回里应该有 drive / lit_spike 等字段
```

最后浏览器打开，确认三件事：

1. **状态栏**显示"冻结脑 144,837 神经元 / 15.02M 突触"
2. **右侧 3D 脑图**有彩色点（LC4 橙、LPLC2 紫、DNp01 红）在闪
3. **鸟会自己飞**（不是一直掉）—— 分数会慢慢涨

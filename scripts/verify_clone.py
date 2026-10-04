"""验证一次"全新 clone"检出的资产真的可用（模拟部署到 Linux 后的状态）。"""
import json
import pathlib
import sys

import numpy as np

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
ok = True


def check(label, fn):
    global ok
    try:
        msg = fn()
        print(f"  ✅ {label:22} {msg}")
    except Exception as e:                       # noqa: BLE001
        ok = False
        print(f"  ❌ {label:22} {type(e).__name__}: {e}")


def j():
    m = json.loads((root / "data" / "spiking_full.json").read_text(encoding="utf-8"))
    return f"n_neurons={m['n_neurons']} n_synapses={m['n_synapses']} keys={len(m)}"


def npz():
    with np.load(root / "data" / "spiking_full.npz") as d:
        ks = sorted(d.files)
        return f"{ks} node_ids={d['node_ids'].shape} indices={d['indices'].shape}"


def coords():
    with np.load(root / "data" / "coords.npz") as d:
        ks = sorted(d.files)
        keys = [k for k in ks if k.startswith("key_")]
        sides = [k for k in ks if k.startswith("side_")]
        return (f"{len(ks)} 项；all={d['all'].shape} lit={d['lit'].shape} "
                f"关键群={len(keys)} 逐侧={len(sides)}")


def three():
    p = root / "demo" / "vendor" / "three.module.js"
    n = p.stat().st_size
    tail = p.read_text(encoding="utf-8", errors="replace")[-60000:]
    assert "export {" in tail, "文件尾部没有 export，可能被行尾转换截断"
    return f"{n:,} B，且含 export 语句"


def src():
    names = ["demo/server.py", "demo/index.html", "demo/app.js",
             "src/fpv/looming.py", "src/fpv/spiking_brain.py",
             "scripts/bake_coords.py"]
    missing = [n for n in names if not (root / n).is_file()]
    assert not missing, f"缺文件: {missing}"
    return f"{len(names)} 个关键文件都在"


print(f"检出的代码根目录: {root.resolve()}")
for label, fn in [("spiking_full.json", j), ("spiking_full.npz", npz),
                  ("coords.npz", coords), ("three.module.js", three),
                  ("关键源码文件", src)]:
    check(label, fn)

# 关键群计数必须与 demo 点亮的那 556 个一致
try:
    with np.load(root / "data" / "coords.npz") as d:
        want = {"LC4": 104, "LPLC2": 210, "DNp01": 2, "DNp04": 2,
                "LC10a": 234, "DNp02": 2, "DNp11": 2}
        got = {k[4:]: int(d[k].size) for k in d.files if k.startswith("key_")}
        same = got == want
        print(f"  {'✅' if same else '❌'} 关键群计数{'一致' if same else '不一致'}"
              f"  {got}")
        ok = ok and same
except Exception as e:                           # noqa: BLE001
    ok = False
    print(f"  ❌ 关键群计数   {e}")

print("\n" + ("全部通过 ✅" if ok else "有问题 ❌"))
sys.exit(0 if ok else 1)

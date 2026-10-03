// JS 语法检查（批量、带清晰输出）。
//
// 血的教训（2026-10-03）：
//   我在 drawPipeSprite 的文档注释里多写了一个 `*/`，于是紧跟的 `*  @param ...`
//   暴露在注释外面。**浏览器直接 SyntaxError、整个模块不执行、页面全黑**，
//   而且因为模块压根没跑，连 app.js 自己的 try/catch 都来不及显示错误 ——
//   页面就永远停在初始文案"载入脑组织中…"，看起来像服务器挂了。
//
//   ⚠️ 更正一个错误说法：`node --check` **是能**抓到这种错的（实测退出码 1、
//   报同样的 Unexpected token '*'）。当时之所以没抓到，是因为**我在加了那段
//   @param 注释之后没有重新跑检查**。所以纪律比工具重要：
//       改完 js 立刻 node --check
//   这个脚本只是让"一次查多个文件 + 输出更清楚"更顺手。
//
// 用法:
//   node demo/check_syntax.mjs demo/app.js demo/verify_pipe_visual.js
//   （带 --experimental-vm-modules 时额外用 ESM 规则解析一遍，更贴近浏览器）
import fs from 'node:fs';
import path from 'node:path';

const files = process.argv.slice(2);
if (!files.length) {
  console.error('用法: node demo/check_syntax.mjs <file.js> [...]');
  process.exit(2);
}

let bad = 0;
let vm = null;
try { vm = await import('node:vm'); } catch (_) {}

for (const f of files) {
  const p = path.resolve(f);
  const src = fs.readFileSync(p, 'utf8');
  try {
    if (vm && vm.SourceTextModule) {
      // 按 ESM 规则解析（只构造、不链接/求值，所以 import 'three' 不受影响）
      new vm.SourceTextModule(src, { identifier: p });
    } else {
      // 没有 --experimental-vm-modules 时的退化路径：把顶层 import/export 摘掉
      // 再当脚本体解析。不这么做的话 `import ... from 'three'` 会误报
      // "Cannot use import statement outside a module"（踩过）。
      const script = src
        .replace(/^\s*import\s[^;\n]*;?\s*$/gm, '')
        .replace(/^\s*export\s+default\s+/gm, 'void 0, ')
        .replace(/^\s*export\s+/gm, '');
      new Function(script);
    }
    console.log(`OK    ${f}`);
  } catch (e) {
    bad++;
    console.log(`FAIL  ${f}: ${e.message}`);
  }
}
console.log(bad === 0 ? '\n全部通过' : `\n${bad} 个文件解析失败`);
console.log('提示：用 `node --experimental-vm-modules demo/check_syntax.mjs <f>` 可按真正的 ESM 规则再查一遍');
process.exit(bad === 0 ? 0 : 1);

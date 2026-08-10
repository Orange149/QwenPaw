# QwenPaw 框架开销测试工具

这套工具用于测量 **QwenPaw 2.0.1 框架本身** 的磁盘、常驻内存、CPU、启动、编排延迟、Prompt、状态写入和进程生命周期成本。主要测试使用远程模型或本地确定性 mock；mock 不是本地模型，也不包含模型推理开销。

目标环境是最终可迁移到 4 核、4 GB、无图形界面的 ARM Linux。当前先在 WSL2 上完成快速筛查。测试不会启动 `qwenpaw app`，也不把 `doctor` 的 8088 健康检查计入结果。

## 安全边界

工具把 `~/.qwenpaw` 和加密 Provider 配置仅作为只读 seed。每个样本都会创建独立的 `/tmp/qwenpaw-overhead.<run-id>...`：临时根目录权限为 `0700`，Secret 文件权限为 `0600`，并重定向 `QWENPAW_WORKING_DIR`、`QWENPAW_SECRET_DIR`、`PAW_STATE_DIR`、`HOME` 和 XDG 缓存目录。因此测试产生的 Session、Memory、日志和缓存不会写回真实的 `~/.qwenpaw`。

还有以下保护：

- 核心场景从空临时项目目录启动，不使用当前源码目录作为 Agent 工作区。
- 子进程环境会清除大小写形式的代理变量，CPU 默认固定在 0–3。
- S2 只接受 QwenPaw 2.0.1 的固定审批卡，并且只自动批准完全匹配的 `printf QWENPAW_BENCH_42`。审批前的 Guard 泛化请求也必须精确指向这个命令，mock 只返回原命令、不扩大授权范围；任何协议或命令偏差都会令样本失败。
- Mock 和 relay 只把脱敏后的请求结构写入结果，不记录 `Authorization`、API Key 或原始 Provider 配置。
- DashScope relay 只允许配置的阿里云 HTTPS 兼容端点，并在本地硬性限制真实请求尝试次数。Token 只能在服务端响应后按权威 usage 结算；达到阈值后停止后续请求，但最后一个请求可能使累计值略超阈值，因此控制台的“免费额度用完即停”仍是计费安全边界。
- 每轮结束会检查 ACP、Shell、浏览器和 MCP 后代是否退出；结果目录最终执行 Secret 扫描，命中后整轮标记失败。
- 临时目录只有带本轮安全标记且路径验证通过后才会删除。进程被强制中断时可能留下本轮目录，应核对准确路径和安全标记后再处理，切勿对 `/tmp` 或 home 目录做递归清理。

工具会读取 seed 配置并在 `/tmp` 中复制一份，因此运行用户必须本来就有权读取这些文件。请勿把 `benchmark_results/` 或残留临时目录分享给不可信用户，即使 Secret 扫描已经通过。

## 前置条件

从仓库根目录执行命令。默认口径固定为：

```text
QwenPaw: 2.0.1
Python:  /home/orange/.qwenpaw/venv/bin/python
Seed:    /home/orange/.qwenpaw
Secret:  /home/orange/.qwenpaw.secret
```

需要 Linux `/proc`、`/proc/<pid>/smaps_rollup`、CPU affinity、`taskset` 和 `psutil`。cgroup v2 可提供补充计数；WSL 未开放相应信息时会降级并在结果中注明，不应伪造缺失值。TUI 测量使用 PTY，仍然不需要桌面或浏览器。

先运行只读前置检查：

```zsh
cd ~/code/QwenPaw
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead preflight
```

检查会确认版本、默认 Agent、加密 Provider seed、可用 CPU、cgroup v2 和已有 QwenPaw 进程。`fail` 必须先解决；`warn` 可以继续，但需要在报告中解释。例如已有 QwenPaw 进程会污染 WSL 系统基线。

所有命令及当前参数以帮助为准：

```zsh
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead --help
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead run --help
```

## CLI 用法

### 1. 静态体积

```zsh
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead footprint
```

它统计 allocated bytes、apparent bytes、文件数、发行包 `RECORD`、原生 `.so` 架构，以及安装、状态、日志、缓存、Secret 和本地模型等层级。默认 ARM 检查只记录将要执行的 pip 命令，不访问网络；显式执行 aarch64/cp312 wheel 解析时使用：

```zsh
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead footprint --arm-dry-run
```

ARM dry-run 可能访问 Python 包索引，只证明依赖在目标 platform tag 下可解析，不证明程序能在真实板卡运行。

### 2. 默认完整离线矩阵

```zsh
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead run --phase all
```

`all` 默认执行静态体积、60 秒 WSL 基线、启动与 90 秒空闲、mock 请求矩阵和一次 stock TUI 探针；**不会自动调用 DashScope**。受控 Profile 为：

| Profile | 主要差异 |
|---|---|
| `full` | 默认 Skills、Tools、ReMeLight 和 Guard |
| `no_skills` | 在 `full` 上禁用全部 Skills |
| `local_tools` | 仅保留三个 builtin 工具；ReMeLight 仍会动态注入 `memory_search`，以便下一步单独测 Memory 消融 |
| `core` | Memory `none`，关闭额外服务，只保留 default Agent、Guard 和恰好三个 wire 工具 |
| `stock_reference` | 保留当前配置和模型参数，仅作为真实体验参考 |

四个受控 Profile 统一使用 `qwen3.7-plus`、`thinking=false`、`temperature=0`、`max_tokens=128` 和流式输出。默认每个受控 Profile 启动 3 次；S1、S2 各运行 3 次；`full`、`core` 另各运行一次 10 轮 S3。每个 mock S1/S2 还会把捕获到的同一 wire request 直接重放给 mock，以计算纯编排税。

可单独运行阶段：

```zsh
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead run --phase static
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead run --phase startup
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead run --phase mock
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead run --phase tui
```

使用 `--profiles` 可以缩小受控 Profile 范围。例如只测核心 Runtime：

```zsh
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead run --phase mock --profiles core
```

不要手工拼接不同 run-id、不同 Profile 或修改过时长的结果来冒充同一组重复样本。每次命令都会建立新的结果目录；若需要一份可直接下结论的完整报告，应在一次 `--phase all` 中同时完成离线和远端阶段。

### 3. Smoke 验收

正式采样前先执行缩短版：

```zsh
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead run --smoke --phase all
```

Smoke 用于验证隔离目录、ACP Ready、mock 流、固定 Shell 工具协议、资源采样、退出检查、产物生成和 Secret 扫描。它会缩短空闲时间和重复次数，**不能**用于对照 4 GB 端侧阈值或给出稳定性能结论。

### 4. DashScope 真实模型矩阵

远端阶段不会仅因配置中已有 API Key 而意外开始。必须同时给出两个确认参数；推荐把远端与离线矩阵放在同一个 run-id 中：

```zsh
read -rs "DASHSCOPE_API_KEY?DashScope API Key: "
echo
export DASHSCOPE_API_KEY

/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead run \
  --phase all \
  --allow-remote-api \
  --confirm-free-quota-stop

unset DASHSCOPE_API_KEY
```

如果只想排查远端协议，也可以把 `--phase all` 改成 `--phase dashscope`，但该目录缺少静态、启动、空闲和 mock 数据，报告只能给出局部结论。

建议第一次真实调用先只测不含工具和 Guard 辅助调用的 S1。该矩阵执行 `direct/full/core × 3` 加一次 stock reference，共 10 次模型请求：

```zsh
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead run \
  --phase dashscope \
  --remote-scenarios S1 \
  --allow-remote-api \
  --confirm-free-quota-stop
```

两个确认的含义分别是：你允许产生真实网络请求；你已经在阿里云控制台开启“免费额度用完即停”或接受潜在计费。Harness 无法替你开启或验证控制台策略。`DASHSCOPE_API_KEY` 是 direct 基线所必需的；不要把 Key 直接写在命令行、README、结果目录或 shell 历史中。

远端矩阵按轮换顺序执行 `direct/full/core × S1/S2 × 3`，再执行一次 `stock_reference/S1`。direct 的 S2 使用相同 Tool Schema，并在客户端执行同一个固定 `printf` 两轮协议。relay 按真正开始的上游 HTTP 请求计数；30 次尝试是请求前的硬限制。服务端总 Token 在每次响应后结算，达到 500,000 后停止后续请求；由于无法预知下一个响应的 usage，最后一个请求可能让累计值越过 500,000：

```text
最多 30 次真实模型尝试
最多 500,000 个服务端总 Token
```

这些上限是额外保险，不代表调用免费，也不能替代阿里云控制台的额度策略。若某次真实尝试没有返回服务端 usage，Harness 会关闭后续远端请求。开始前还应确认 seed 的 Provider URL 和模型确实是预期的 DashScope 配置。远端失败不应通过自动无限重试“补齐”样本。

QwenPaw 2.0.1 的 Guard 在每个需要审批的 S2 样本中还会产生一次目标泛化模型调用。它会如实计入 30 次上限，因此严格上限可能让矩阵末尾的 `core/S2` 或 `stock_reference/S1` 被记录为 budget skip；Harness 不会偷偷提高额度来补齐。
Budget skip 不计作一次真实正确性试验；任何 full/core × S1/S2 分组少于 3 次实际执行时，远端正确性结论保持 unknown。Mock 的主机框架延迟门限只使用无辅助模型调用的 S1；S2 报告的是包含 Guard 的整体框架路径额外延迟。

### 5. 重建报告

已有原始产物时可以重新聚合，不重新启动 QwenPaw 或调用模型：

```zsh
/home/orange/.qwenpaw/venv/bin/python -m benchmarks.overhead report benchmark_results/<run-id>
```

每组只有 3 个独立样本，因此报告只给原始值、中位数和范围。p95 只用于具有足够时间点的 90 秒空闲序列。

## 场景定义

- **S1**：从约 256 Token 的固定合成文本中读取校验码，禁用工具，只输出短答案。
- **S2**：只允许一次 `execute_shell_command`，命令严格为 `printf QWENPAW_BENCH_42`。
- **S3**：同一 Session 连续 10 轮固定键值，用于观察历史增长、PSS、FD、线程和状态文件增量。

Mock 返回确定性短文本或一次确定性工具调用，用来隔离框架开销。自动标题、自动 Memory、backend warmup 和 Guard 审批泛化会被严格识别为辅助模型调用：它们计入尝试、Token 和资源成本，但不会打乱 S1/S2/S3 的主请求轮次。Mock usage 是合成计数，只用于协议和内部交叉检查，不能当作服务端权威 Token；DashScope 的 direct 与 QwenPaw 路径才用于分解真实网络和模型波动下的额外成本。

## 结果产物

默认写入 `benchmark_results/<run-id>/`：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 版本、主机、参数、配置哈希、运行状态、VmmemWSL 边界旁证和完成时间；不含原始配置 |
| `footprint.csv` | 目录/发行包/原生库的 allocated、apparent、文件数和 ARM 信息 |
| `samples.csv` | 按时间采样的进程树资源数据，区分 ACP、TUI 和工具子进程 |
| `events.jsonl` | ACP、mock/relay、工具、Ready、Turn End 等时间事件 |
| `requests.jsonl` | 脱敏后的请求结构、usage 和 Prompt 近似分解，不保存 Authorization |
| `measurements.jsonl` | 每个独立样本的长表记录、成功状态、错误和聚合指标 |
| `summary.csv` | 按 profile/scenario/backend/metric 聚合的中位数与范围 |
| `report.md` | 阈值判定、裁剪候选、ARM 阻塞和局限性 |
| `raw_samples/` | 每个样本的原始资源采样，便于定位异常值 |

`manifest.json` 的 `status=completed` 且 Secret finding 为零，才表示产物通过收尾检查。`partial`、失败样本和 `unknown` 指标必须保留，不要当作零值。离线或 smoke 结果会明确标为 provisional，缺少每组 3 次正式样本、DashScope 成功率或未来最小包实测值时不会给出伪造的最终通过结论。建议先看 `report.md`，再用 `summary.csv` 对照原始 `measurements.jsonl` 和 `samples.csv`。

Socket 端点来自周期性的 `/proc` 进程树采样，是“观测到的连接”，不是数据包级完整审计；relay 另行记录配置的上游域名并执行固定 host allowlist。cgroup v2 数据是共享层级的交叉校验，不能冒充 QwenPaw 独占值。需要不可绕过的网络白名单时，应在 ARM 板复测阶段增加 network namespace 或防火墙规则。

核心阈值围绕 4 核/4 GB 端侧设定，包括：部署包、空闲 PSS/CPU、Runtime Ready、mock 编排税、固定 Prompt、10 轮资源增长、后代退出、任务正确性和 ARM wheel 可解析性。当前完整 venv 体积是安装上界，不等同于“未来最小部署包”；真正的最小包需要在裁剪并重新打包后重新测量。

## TUI 与 Coding Mode 注意事项

核心 Runtime 的 Ready 以 ACP `AvailableCommands` 为准。TUI 状态栏的 `ready` 是默认 backend warmup 后的 UI 状态，两者不是同一时间点。

QwenPaw 官方 TUI 会把项目路径作为 metadata 传给 ACP；即使传入的是空临时目录，也可能自动启用 Coding Mode。因此 `stock_reference` TUI 结果刻意单列，包含 Textual 父进程、ACP 子进程、默认 warmup、定时器以及 Coding/Plan 上下文成本，不能直接当成 `core` 无头 Runtime。此工具不启动 Web Console，也不安装 Playwright/Chromium。

## WSL2 与 ARM 解释限制

WSL2 结果适合做相对消融，但不是板级结论：

- 不清 Linux Page Cache，所以启动结果叫“新进程启动”，不叫真正冷启动。
- PSS/USS 和进程 CPU-s 来自 WSL Linux 侧；Windows 的 `VmmemWSL` 是整个 VM，不能准确归因到单个 QwenPaw 样本。
- WSL 的调度、虚拟磁盘、网络转发和内存回收会影响结果。磁盘 free 表示 WSL 文件系统所在卷的可用空间，不表示 QwenPaw 会覆盖 Windows 文件。
- 当前 Harness 自动报告 Linux 侧 CPU-s、时间和内存，并在整轮计时区间之外尽力读取 Windows `VmmemWSL` 的前后 Working Set；读取失败记为 unavailable，不填零。该增量只能作为整个 WSL VM 的旁证，不能归因成单进程 PSS，更不能换算成焦耳。第一轮也不测试温度、闪存写放大或长时间空闲。
- DashScope TTFT/E2E 受公网和服务端负载影响，应依靠轮换顺序与 direct 对照，不应把 3 次样本解释成稳定分布。

aarch64/cp312 dry-run 只检查当前顶层 requirement 在目标 tag 下的解析，报告会标出 resolver timeout、缺 wheel/无匹配版本等阻塞；它不能证明所有运行时可选路径、目标 libc 或板卡程序都可用。通过快速筛查后，应把同一 harness 放到目标 ARM 板，确认实际发行版和 libc，重新设置可用 CPU affinity，并完成 100 次稳定性、板级功耗、温度、闪存写放大及长期空闲测试。缺少 ARM wheel 的必需原生依赖应先视为移植阻塞项，而不是用 WSL x86_64 成绩绕过。

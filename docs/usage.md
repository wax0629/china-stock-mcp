# china-stock-mcp 使用指南

> 面向中文金融市场（A 股 / 港股 / 公募基金）的 Model Context Protocol 服务器。

本文档面向最终使用者，介绍如何安装 `china-stock-mcp`、如何在主流 MCP 客户端（Claude Desktop / Cursor / Cline）中接入它，以及如何调用它提供的 10 个 Tool、3 个 Prompt 和 3 个 Resource。

如果你是开发者并希望了解内部架构、错误处理、缓存与限流细节，请参阅 [`design.md`](../.kiro/specs/china-stock-mcp/design.md) 与 [`requirements.md`](../.kiro/specs/china-stock-mcp/requirements.md)。

---

## 目录

1. [产品边界](#产品边界)
2. [快速开始](#快速开始)
3. [安装](#安装)
4. [环境变量](#环境变量)
5. [接入 MCP 客户端](#接入-mcp-客户端)
6. [Tools 速查表](#tools-速查表)
7. [Tool 详细说明](#tool-详细说明)
8. [Prompts](#prompts)
9. [Resources](#resources)
10. [错误处理](#错误处理)
11. [常见问题](#常见问题)

---

## 产品边界

服务严格遵守「**不下单、不荐股、不做实时 L1/L2、不收集 PII**」的边界：

- **不下单**：不暴露任何下单 / 委托 / 交易接口。
- **不荐股**：所有输出仅供研究学习使用，不构成投资建议。
- **数据有延迟**：行情数据来自公开第三方，约有 15 分钟延迟。
- **隐私保护**：日志只记录工具名 / 时间戳 / 缓存命中状态，不记录用户身份 / 持仓 / query 内容。
- **密钥仅从环境变量读取**：`CSM_TUSHARE_TOKEN` 等不会被写入磁盘缓存或日志。

每条 Tool / Prompt / Resource 输出末尾固定追加：

> ⚠️ 数据来源于公开第三方，可能存在延迟或误差。本服务仅供研究学习使用，不构成任何投资建议。

---

## 快速开始

最快的方式是用 `uvx` 启动：

```bash
uvx china-stock-mcp
```

然后在你使用的 MCP 客户端里加一段配置，让它把 `china-stock-mcp` 拉起来。三个最常见的客户端示例：

**Claude Desktop**（JSON）：

```json
{
  "mcpServers": {
    "china-stock": {
      "command": "uvx",
      "args": ["china-stock-mcp"]
    }
  }
}
```

**OpenAI Codex CLI**（TOML，文件 `~/.codex/config.toml`）：

```toml
[mcp_servers.china-stock]
command = "uvx"
args = ["china-stock-mcp"]
enabled = true
```

**Cursor**：在设置 → MCP 中加入与 Claude Desktop 相同结构的 JSON。

重启对应客户端后，即可在对话中提问：

> 帮我分析一下宁德时代当前的基本面和近期资金流向。

模型会自动调用 `search_symbol` → `get_fundamentals` → `get_money_flow` 等工具完成分析。

更多客户端的接入方式见下面的 [接入 MCP 客户端](#接入-mcp-客户端) 章节。

---

## 安装

要求 **Python 3.11 或更高版本**。

### 方式一：uvx 一键运行（推荐）

无需手动安装，每次自动拉取最新版本：

```bash
uvx china-stock-mcp
```

启用所有备用数据源：

```bash
uvx "china-stock-mcp[all]"
```

### 方式二：从 PyPI 安装到全局环境

```bash
pip install "china-stock-mcp[all]"
china-stock-mcp
```

可选 extras：

| Extra | 作用 |
| ----- | ---- |
| `[tushare]` | 启用 tushare 备用数据源（需同时设置 `CSM_TUSHARE_TOKEN`） |
| `[efinance]` | 启用 efinance 备用数据源 |
| `[redis]` | 启用 redis 缓存后端 |
| `[all]` | 同时安装上述全部 extras |

### 方式三：本地开发

```bash
git clone https://github.com/wax0629/china-stock-mcp
cd china-stock-mcp
uv sync --all-extras --group dev
uv run china-stock-mcp
```

---

## 环境变量

所有环境变量统一以 `CSM_` 前缀。

| 变量 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `CSM_CACHE_BACKEND` | `disk` | 缓存后端：`disk`（diskcache）或 `redis` |
| `CSM_CACHE_DIR` | `~/.cache/china-stock-mcp` | diskcache 根目录（仅 `disk` 后端使用） |
| `CSM_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `CSM_TUSHARE_TOKEN` | _未设置_ | 可选 tushare API token；用于启用 tushare 备用源 |
| `CSM_RATE_LIMIT` | `30` | Token Bucket 全局限流容量（每分钟最大请求数） |
| `CSM_DATA_DELAY_NOTICE` | `true` | 是否在行情响应中标注「数据延迟约 15 分钟」 |
| `CSM_TRANSPORT` | `stdio` | 传输层：`stdio` 或 `streamable-http` |

布尔值接受 `true/false/yes/no/1/0/on/off`，大小写不敏感。

> **关于日志**：`stdio` 传输模式下，MCP 协议帧走 stdout，日志走 stderr，二者互不干扰。

---

## 接入 MCP 客户端

`china-stock-mcp` 是标准的 MCP 服务器，**任何兼容 MCP 协议的客户端**都能接入，不限于 Claude Desktop。常见客户端见下表（按字母序）：

| 客户端 | 配置格式 | 配置文件位置 |
| ------ | -------- | ------------ |
| Claude Code（CLI） | JSON | `~/.claude/mcp.json` |
| Claude Desktop | JSON | macOS: `~/Library/Application Support/Claude/claude_desktop_config.json` / Windows: `%APPDATA%\Claude\claude_desktop_config.json` |
| Cline (VS Code 扩展) | JSON | 设置面板 → MCP Servers |
| Continue (VS Code 扩展) | JSON | `~/.continue/config.json` |
| Cursor | JSON | 设置 → MCP |
| Goose（Block 出品的 CLI agent） | YAML | `~/.config/goose/config.yaml` |
| OpenAI Codex CLI / IDE 扩展 | **TOML** | `~/.codex/config.toml` |
| Zed 编辑器 | JSON | 设置面板 |

各客户端在协议层等价，只是配置文件格式与字段名略有差异。下面给出最常用的几个示例。

### Claude Desktop

#### 最简配置

```json
{
  "mcpServers": {
    "china-stock": {
      "command": "uvx",
      "args": ["china-stock-mcp"]
    }
  }
}
```

#### 启用 tushare 备用源

```json
{
  "mcpServers": {
    "china-stock": {
      "command": "uvx",
      "args": ["china-stock-mcp[tushare]"],
      "env": {
        "CSM_TUSHARE_TOKEN": "your_tushare_token_here",
        "CSM_CACHE_DIR": "${HOME}/.cache/china-stock-mcp",
        "CSM_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

#### 启用全部备用源 + redis 缓存

```json
{
  "mcpServers": {
    "china-stock": {
      "command": "uvx",
      "args": ["china-stock-mcp[all]"],
      "env": {
        "CSM_TUSHARE_TOKEN": "your_tushare_token_here",
        "CSM_CACHE_BACKEND": "redis",
        "CSM_RATE_LIMIT": "60"
      }
    }
  }
}
```

修改完配置后，**完全退出并重新启动 Claude Desktop** 才能生效。

### OpenAI Codex CLI / IDE

Codex 使用 TOML 配置，CLI 与 IDE 扩展共享同一份配置：

- 全局：`~/.codex/config.toml`（Windows 下 `%USERPROFILE%\.codex\config.toml`）
- 仓库级：`<repo>/.codex/config.toml`（仅对该仓库生效）

> Codex 目前**只支持 stdio 本地子进程**，不支持远程 MCP。`china-stock-mcp` 默认就是 stdio，无需额外改动。

#### 最简配置

```toml
[mcp_servers.china-stock]
command = "uvx"
args = ["china-stock-mcp"]
enabled = true
```

#### 启用 tushare 备用源

```toml
[mcp_servers.china-stock]
command = "uvx"
args = ["china-stock-mcp[tushare]"]
enabled = true
startup_timeout_sec = 30
tool_timeout_sec = 60

[mcp_servers.china-stock.env]
CSM_TUSHARE_TOKEN = "your_tushare_token_here"
CSM_CACHE_DIR = "${HOME}/.cache/china-stock-mcp"
CSM_LOG_LEVEL = "INFO"
```

> Windows 下 `${HOME}` 不会自动展开，请改用绝对路径如 `C:/Users/你的用户名/.cache/china-stock-mcp`。

#### 命令行管理

```bash
# 直接通过 CLI 添加
codex mcp add china-stock -- uvx china-stock-mcp

# 列出所有已配置的 MCP server
codex mcp list

# 查看具体配置
codex mcp get china-stock

# 移除
codex mcp remove china-stock
```

#### 工具白名单（按需收紧）

如果你只想让 Codex 用到部分工具，可以加 `enabled_tools`：

```toml
[mcp_servers.china-stock]
command = "uvx"
args = ["china-stock-mcp"]
enabled_tools = ["search_symbol", "get_quote", "get_kline"]
```

### Cursor

在 Cursor 设置 → MCP 配置中加入与 Claude Desktop 相同结构的 JSON：

```json
{
  "mcpServers": {
    "china-stock": {
      "command": "uvx",
      "args": ["china-stock-mcp"]
    }
  }
}
```

### Cline / Continue (VS Code 扩展)

打开各自插件的 MCP 配置面板，加入相同结构的 JSON。

### 远程部署（Streamable HTTP）

如果客户端支持 streamable-http（如 Claude Code），可以把服务部署到远程主机：

```bash
CSM_TRANSPORT=streamable-http china-stock-mcp
```

然后按客户端文档接入对应端点。**注意 Codex 当前不支持远程 MCP**，需要本地 stdio。

---

## Tools 速查表

| Tool | 用途 | 关键参数 |
| ---- | ---- | -------- |
| `search_symbol` | 中文名 / 拼音 / 数字代码 → 标准化代码 | `query`, `market` |
| `get_quote` | 单标的或批量行情快照 | `symbol`（字符串或 1–20 项列表） |
| `get_kline` | K 线 + 技术指标 + 形态简评 | `symbol`, `period`, `count`, `adjust`, `indicators` |
| `get_fundamentals` | 估值 / 盈利 / 成长 / 健康四组指标 | `symbol` |
| `get_financial_report` | 多期年报 / 季报 | `symbol`, `report_type`, `periods` |
| `get_money_flow` | 北向 / 主力 / 龙虎榜资金流向 | `symbol`, `flow_type`, `top_n` |
| `get_industry_peers` | 同行业可比公司对比 + 行业分位 | `symbol`, `metrics`, `top_n` |
| `get_fund_info` | 公募基金信息、收益、持仓 | `fund_code`（6 位数字） |
| `screen_stocks` | 多因子选股 | `criteria`, `sort_by`, `order`, `limit` |
| `get_market_overview` | 大盘指数 / 涨跌家数 / 涨跌停 / 北向 / 行业热度 | _无参数_ |

所有 Tool 返回 Markdown 字符串，并自动追加免责声明。

---

## Tool 详细说明

### `search_symbol`

**用途**：把用户给出的中文名、拼音、不带后缀的数字代码统一为标准化代码（如 `300750.SZ`）。其他 Tool 接受标准化代码或原始输入，不过先调一次 `search_symbol` 能让 AI 看清究竟匹配到了哪个标的。

**参数**：

- `query` _(string, 必填)_：搜索关键字。支持
  - 6 位 A 股代码：自动补 `.SH` / `.SZ` / `.BJ`
  - 5 位港股代码：自动补 `.HK`
  - 6 位基金代码
  - 中文公司名 / 拼音片段
- `market` _(string, 默认 `"all"`)_：市场过滤，取 `"a_stock"` / `"hk_stock"` / `"fund"` / `"all"`。

**示例查询**：

| query | 解释 |
| ----- | ---- |
| `"宁德时代"` | 中文名 |
| `"NDSD"` | 拼音首字母 |
| `"300750"` | 6 位 A 股代码 |
| `"300750.SZ"` | 已标准化 |
| `"00700"` | 港股腾讯 |
| `"510300"` | 沪深 300 ETF |

**返回**：标准化代码、名称、市场、行业的 Markdown 表格。

---

### `get_quote`

**用途**：取近实时（约 15 分钟延迟）行情快照。

**参数**：

- `symbol` _(string 或 string[], 必填)_：单个标准化代码，或最多 20 个标的的列表。

**返回**：

- 单个标的：Markdown 卡片，含价格、涨跌幅、成交量 / 额、换手率、PE_TTM、PE 动态、PB、总市值、流通市值。
- 多个标的：多列 Markdown 表格对比同样的关键字段。

启用 `CSM_DATA_DELAY_NOTICE`（默认）时，会在开头追加「数据延迟约 15 分钟」。

---

### `get_kline`

**用途**：取 K 线 + 技术指标 + 形态简评。

**参数**：

- `symbol` _(string, 必填)_：A 股标准化或裸代码（v1 暂不支持港股 / 基金）。
- `period` _(string, 默认 `"daily"`)_：`"daily"` / `"weekly"` / `"monthly"` / `"60min"` / `"30min"`。
- `count` _(int, 默认 `60`)_：K 线根数，`[1, 250]`。
- `adjust` _(string, 默认 `"qfq"`)_：复权方式，`"qfq"`（前复权）/ `"hfq"`（后复权）/ `"none"`。
- `indicators` _(string[], 默认 `["MA20", "MA60", "MACD"]`)_：从 `MA5` / `MA10` / `MA20` / `MA60` / `MACD` / `RSI14` / `BOLL` 中选择。

**返回**：Markdown 文档，含 K 线根数、日期范围、最新收盘价、最新一根涨跌幅、可选的形态简评（≥ 60 根 K 线时输出「上升趋势」/「下降趋势」/「震荡」）、指标快照表，以及最近 20 根 OHLCV。

---

### `get_fundamentals`

**用途**：取基本面四组指标快照。

**参数**：

- `symbol` _(string, 必填)_：A 股标准化或裸代码。

**返回**：四个 Markdown 子表格

- 估值指标（PE_TTM / PE_动 / PB / PS / PEG）
- 盈利能力（ROE / ROA / 毛利率 / 净利率）
- 成长性（营收同比 / 净利润同比 / 单季环比）
- 财务健康（资产负债率 / 流动比率 / 经营性现金流 / 净利润比）

如果上游返回行业分位数据，会追加「行业分位」列；否则该列省略。

---

### `get_financial_report`

**用途**：取多期年报或季报。

**参数**：

- `symbol` _(string, 必填)_：A 股标准化或裸代码。
- `report_type` _(string, 默认 `"annual"`)_：`"annual"`（年报）或 `"quarterly"`（年报+中报+季报）。
- `periods` _(int, 默认 `4`)_：期数，`[1, 12]`。

**返回**：Markdown 表格，行为指标，列为报告期（按时间升序）。指标包括：营业总收入、归母净利润、扣非净利润、毛利、经营性现金流、总资产、总负债、所有者权益。

> 新股期数不足时，会抛出 `DataNotFoundError`，提示缩小 `periods` 或切换 `report_type`。

---

### `get_money_flow`

**用途**：取资金流向。

**参数**：

- `symbol` _(string | None, 视情况)_：
  - `flow_type="north"`：忽略
  - `flow_type="main"`：必填
  - `flow_type="dragon_tiger"`：可选（提供则按代码过滤）
- `flow_type` _(string, 默认 `"north"`)_：`"north"` / `"main"` / `"dragon_tiger"`。
- `top_n` _(int, 默认 `20`)_：行数上限，`[1, 100]`。

**返回**：Markdown 表格，列结构按 `flow_type` 自适应：

- `north`：日期、净流入、买入、卖出、持股市值
- `main`：日期、主力 / 超大单 / 大单 / 中单 / 小单净流入
- `dragon_tiger`：日期、代码、名称、净买额、买入额、卖出额、换手率、上榜原因

每条响应顶部含 `数据时间: ...` 行。

---

### `get_industry_peers`

**用途**：同行业可比公司对比。

**参数**：

- `symbol` _(string, 必填)_：A 股标准化或裸代码。
- `metrics` _(string[], 默认全部支持的指标)_：从 `"pe"` / `"pb"` / `"roe"` / `"revenue_growth"` 中选择。
- `top_n` _(int, 默认 `10`)_：可比公司数量，`[1, 50]`。

**返回**：Markdown 表格，列为代码 / 名称 + 调用方指定的指标（顺序保留）。表格底部追加每个指标的「行业分位」说明，方便看出标的在行业内的位置。

---

### `get_fund_info`

**用途**：取公募基金信息。

**参数**：

- `fund_code` _(string, 必填)_：6 位裸基金代码（不带交易所后缀，例如 `"510300"`）。非 6 位数字会抛 `SymbolError`。

**返回**：Markdown 文档，含

- 基本信息：名称 / 基金经理 / 成立日期 / 规模 / 近 1/3/6/12 月收益率 / 最大回撤 / 夏普比率 / 同类排名
- 前十大持仓：代码 / 名称 / 权重（2 位小数百分比）
- 行业分布（如有）

缺失字段以 `-` 占位。

---

### `screen_stocks`

**用途**：多因子选股。

**参数**：

- `criteria` _(object | None, 可选)_：因子约束。支持的键：
  - `pe_ttm` / `pb` / `roe` / `market_cap` / `revenue_growth`：取 `{"min": ..., "max": ...}`，两端均可省略
  - `industry`：行业名称列表，多个行业取并集
- `sort_by` _(string, 默认 `"market_cap"`)_：`"pe_ttm"` / `"pb"` / `"roe"` / `"market_cap"` / `"revenue_growth"`。
- `order` _(string, 默认 `"desc"`)_：`"asc"` 或 `"desc"`。
- `limit` _(int, 默认 `30`)_：返回行数，`[1, 200]`。

**示例**：

```json
{
  "criteria": {
    "pe_ttm": {"min": 0, "max": 30},
    "roe":    {"min": 15},
    "industry": ["电池", "新能源车"]
  },
  "sort_by": "market_cap",
  "order": "desc",
  "limit": 50
}
```

**返回**：Markdown 选股结果表，列为代码 / 名称 / 行业 + 调用方实际过滤的指标列。

> **v1 限制**：`roe` 和 `revenue_growth` 在通用选股池中不可得，过滤或排序这两个字段会返回空结果；如需相关数据请改用 `get_industry_peers`。

---

### `get_market_overview`

**用途**：取 A 股市场总览快照。

**参数**：无。

**返回**：Markdown 文档，含

- 数据时间 + 可选「非交易时段」标注
- 指数行情（上证 / 深证成指 / 创业板）：名称 / 代码 / 最新 / 涨跌幅
- 涨跌家数（上涨 / 下跌 / 平）
- 涨跌停数（涨跌幅 ≥ 9.9 视为涨停，≤ -9.9 视为跌停）
- 北向资金净流入
- 行业热度排行（按主力净流入降序，前 5 名）
- 市场热度评分（`XX.X / 100`）

非交易时段返回最近一个交易日的快照，并在顶部追加「非交易时段」横幅。

---

## Prompts

### `research_report` — 投研报告

把基本面、财务、行业对比、资金流向、技术形态串成一份完整的投研报告。

**参数**：

- `symbol` _(string, 必填)_：A 股标的。
- `report_length` _(string, 默认 `"standard"`)_：
  - `"short"`：截断到约 1500 token，适合快速浏览
  - `"standard"`：5 段完整报告
  - `"deep"`：在标准基础上追加「深度分析」段落（不会再触发上游调用）

任一子模块失败时，对应段落会替换为「⚠️ 该子模块数据不可用」提示，其他段落正常输出。

### `valuation_compare` — 估值对比

对 2 到 10 只标的做横向估值对比。

**参数**：

- `symbols` _(string[], 必填)_：2–10 只标的，可以混用标准化代码、裸代码、中文名、拼音。

**输出结构**：

1. 行情对比（多列表格）
2. 估值横向对比（每只标的的 PE / PB / ROE / 营收增速）
3. 行业横切（每只标的所在行业 + 抽样 peer 数）

### `weekly_review` — 周复盘

针对 A 股市场的周复盘报告。

**参数**：无。

**输出结构**：

1. 市场总览
2. 北向资金近 20 个交易日走势
3. 行业热度排行

---

## Resources

Resources 是 MCP 中的「订阅式」资源，调用方式由客户端决定（Claude Desktop 通常以"附件"形式呈现给模型）。

| URI | 说明 |
| --- | ---- |
| `market://overview` | 市场总览（与 `get_market_overview` 同源） |
| `flow://north` | 北向资金（等价于 `get_money_flow(flow_type="north", top_n=20)`） |
| `symbol://{code}/profile` | 标的画像；`{code}` 可填标准化代码、裸代码、中文名、拼音 |

---

## 错误处理

所有错误均经过统一映射，**不会泄露 Python traceback**。返回给客户端的消息为人类可读的中文摘要，例如：

| 异常类型 | 触发场景 | 用户消息示例 |
| -------- | -------- | ----------- |
| `SymbolError` | 无法识别的代码 / 中文名 | `无法识别的标的: 'ABCDE'。是否指: 宁德时代 / 比亚迪 / 三一重工?` |
| `ValidationError` | 参数越界 | `参数 periods 必须在 [1, 12] 范围内, 当前值 25。` |
| `DataNotFoundError` | 上游无数据 | `300999.SZ 暂无 12 期年报, 请缩小 periods 或切换 report_type=quarterly。` |
| `RateLimitError` | 触发限流 | `请求频率超限, 请稍后重试。` |
| `NetworkError` | 网络异常 | `数据源连接失败, 已自动尝试备用源仍失败。` |
| `DataSourceError` | 数据格式异常 | `数据源返回格式异常, 请稍后重试。` |

主源失败时，服务会按以下规则尝试备用源（仅在 `NetworkError` / `RateLimitError` 切换；`DataNotFoundError` 不切换以避免假阳性）：

| Service | 主源 | 备用源 |
| ------- | ---- | ------ |
| 标的搜索 | akshare | tushare → efinance |
| 行情快照 | akshare | efinance → tushare |
| 基本面 / 财务 | akshare | tushare |
| 资金流向 | akshare | efinance |
| K 线 / 行业 / 基金 / 选股 / 市场总览 | akshare | _无_ |

---

## 常见问题

**Q：装好了但客户端看不到工具。**

A：检查：
1. 配置文件语法正确（JSON 用 `python -m json.tool < ...config.json` 验证；TOML 可用 `python -c "import tomllib; tomllib.load(open('config.toml','rb'))"`）。
2. 完全退出并重启客户端（Claude Desktop / Cursor / Codex IDE 等需要重启进程才会重读配置）。
3. 查看客户端的 MCP 启动日志，常见错误是 `uvx not found` —— 把 `uvx` 改成绝对路径（`which uvx` 或 Windows 下 `where uvx` 取得）。
4. Codex 用户可以跑 `codex mcp list` 与 `codex mcp get china-stock` 直接确认配置被读到。

**Q：tushare 备用源没有生效。**

A：必须同时满足：
1. 装了 `china-stock-mcp[tushare]` 或单独安装了 `tushare` 包。
2. 设置了 `CSM_TUSHARE_TOKEN` 环境变量。
3. 启动日志中能看到 `services wired: tushare=True`。

**Q：能不能取实时 L1/L2 行情？**

A：不能。本服务严格遵守「不做实时 L1/L2」的边界，行情数据来自公开第三方，约有 15 分钟延迟。

**Q：rate limit 默认 30 req/min 是不是太低？**

A：保守默认是为了不打爆 akshare。如果你部署在自己的服务器上、流量大，可以设 `CSM_RATE_LIMIT=120` 或更高。注意 token bucket 是全局的，不是按工具或按用户分的。

**Q：磁盘缓存能清吗？**

A：直接 `rm -rf ~/.cache/china-stock-mcp`（或 `CSM_CACHE_DIR` 指向的目录）即可，缓存会在下次访问时按需重建。

**Q：怎么调试？**

A：
1. 设置 `CSM_LOG_LEVEL=DEBUG`，重启服务。
2. stdio 模式下，日志走 stderr（Claude Desktop 把 stderr 写到 `~/Library/Logs/Claude/` 或 `%APPDATA%\Claude\logs\`）。
3. 直接运行 `china-stock-mcp` 在终端中复现，stderr 会直接输出到屏幕。

**Q：能离线用吗？**

A：能查的范围限于已经被缓存的请求。第一次取数据需要联网。

---

## 参考

- [`design.md`](../.kiro/specs/china-stock-mcp/design.md) — 内部架构、算法、Correctness Properties
- [`requirements.md`](../.kiro/specs/china-stock-mcp/requirements.md) — 功能需求与验收标准
- [Model Context Protocol 官方文档](https://modelcontextprotocol.io/)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [akshare](https://github.com/akfamily/akshare)

---

> ⚠️ 数据来源于公开第三方，可能存在延迟或误差。本服务仅供研究学习使用，不构成任何投资建议。

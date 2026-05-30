# china-stock-mcp

> 面向中文金融市场（A 股 / 港股 / 公募基金）的 Model Context Protocol 服务器。

`china-stock-mcp` 让任何兼容 MCP 的 AI 客户端（Claude Desktop / Cursor / Cline 等）通过统一接口取得"懂中国市场"的研究分析能力：标的搜索、行情快照、K 线与技术指标、基本面与财务、资金流向、行业对比、公募基金、多因子选股、市场总览，外加三个可直接调用的研究类 Prompt（投研报告 / 估值对比 / 周复盘）。

服务以 [FastMCP](https://github.com/jlowin/fastmcp) 为协议层，下层通过统一的 Adapter 抽象封装 [akshare](https://github.com/akfamily/akshare)（主源）/ [tushare](https://tushare.pro)（备用）/ [efinance](https://github.com/Micro-sheep/efinance)（备用），并通过 [diskcache](https://github.com/grantjenks/python-diskcache)（默认）或 redis（可选）做 TTL 分级缓存与 Token Bucket 全局限流。

---

## 产品边界（Product Scope）

> 本服务严格遵循「**本地隐私优先、不下单、不荐股、不做实时 L1/L2**」的产品边界。

- **不下单**：服务不暴露任何下单 / 委托 / 交易接口；所有工具仅返回数据或计算结果。
- **不荐股**：所有输出仅供研究学习使用，不构成任何投资建议。
- **不做实时 L1/L2**：行情数据来自公开第三方，约有 15 分钟延迟。
- **不收集 PII**：不记录用户身份 / 持仓 / 交易记录，仅记录工具名、时间戳、缓存命中状态。
- **密钥仅从环境变量读取**：`CSM_TUSHARE_TOKEN` 等不会被写入磁盘缓存或日志。

每条工具 / Prompt / Resource 返回的 Markdown 末尾固定追加：

> ⚠️ 数据来源于公开第三方，可能存在延迟或误差。本服务仅供研究学习使用，不构成任何投资建议。

---

## Features

### Tools（10 个）

| Tool | 说明 |
| ---- | ---- |
| `search_symbol` | 中文名 / 拼音 / 数字代码 → 标准化代码（A 股 / 港股 / 公募基金） |
| `get_quote` | 单标的或批量（最多 20 个）行情快照，含估值与成交数据 |
| `get_kline` | K 线 + MA / MACD / RSI / BOLL 等技术指标 + 形态简评 |
| `get_fundamentals` | 估值 / 盈利 / 成长 / 健康四组基本面指标 + 行业分位 |
| `get_financial_report` | 多期年报 / 季报，可选 1–12 期 |
| `get_money_flow` | 北向 / 主力 / 龙虎榜资金流向 |
| `get_industry_peers` | 同行业可比公司对比 + 行业分位说明 |
| `get_fund_info` | 公募基金信息、收益、持仓与行业分布 |
| `screen_stocks` | 多因子选股（PE / PB / ROE / 市值 / 成长 / 行业） |
| `get_market_overview` | 大盘指数 / 涨跌家数 / 涨跌停 / 北向 / 行业热度 / 热度评分 |

### Prompts（3 个）

| Prompt | 说明 |
| ------ | ---- |
| `research_report` | 投研报告：基本面 + 财务 + 行业对比 + 资金流向 + 技术形态 |
| `valuation_compare` | 估值对比：组合行情 + 基本面 + 行业对比 |
| `weekly_review` | 周复盘：市场总览 + 北向资金 + 行业涨跌 |

### Resources（3 个）

- `market://overview` — 市场总览
- `market://north-flow` — 北向资金流向
- `symbol://{code}/profile` — 标的画像

---

## Installation

要求 **Python 3.11+**。

### 方式一：uvx 一键运行（推荐）

```bash
uvx china-stock-mcp
```

### 方式二：从 PyPI 安装

```bash
pip install "china-stock-mcp[all]"
china-stock-mcp
```

可选 extras：

- `[tushare]` — 启用 tushare 备用数据源
- `[efinance]` — 启用 efinance 备用数据源
- `[redis]` — 启用 redis 缓存后端
- `[all]` — 上述全部

### 方式三：本地开发

```bash
git clone https://github.com/wax0629/china-stock-mcp
cd china-stock-mcp
uv sync --all-extras --group dev
uv run china-stock-mcp
```

---

## Environment Variables

所有环境变量统一以 `CSM_` 前缀。详细语义见 [`config.py`](src/china_stock_mcp/config.py)。

| 变量 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `CSM_CACHE_BACKEND` | `disk` | 缓存后端：`disk`（diskcache）或 `redis` |
| `CSM_CACHE_DIR` | `~/.cache/china-stock-mcp` | diskcache 根目录（仅 `disk` 后端使用） |
| `CSM_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `CSM_TUSHARE_TOKEN` | _未设置_ | 可选 tushare API token；用于启用 tushare 备用源 |
| `CSM_RATE_LIMIT` | `30` | Token Bucket 全局限流容量（每分钟最大请求数） |
| `CSM_DATA_DELAY_NOTICE` | `true` | 是否在行情响应中标注「数据延迟约 15 分钟」 |
| `CSM_TRANSPORT` | `stdio` | 传输层：`stdio` 或 `streamable-http` |

> 日志默认写入 stderr，stdio MCP 协议帧走 stdout，二者互不污染。

---

## Claude Desktop Integration

在 `claude_desktop_config.json` 中加入：

```json
{
  "mcpServers": {
    "china-stock": {
      "command": "uvx",
      "args": ["china-stock-mcp"],
      "env": {
        "CSM_CACHE_DIR": "${HOME}/.cache/china-stock-mcp",
        "CSM_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

如需启用 tushare 备用源：

```json
{
  "mcpServers": {
    "china-stock": {
      "command": "uvx",
      "args": ["china-stock-mcp[tushare]"],
      "env": {
        "CSM_TUSHARE_TOKEN": "your_tushare_token_here",
        "CSM_CACHE_DIR": "${HOME}/.cache/china-stock-mcp"
      }
    }
  }
}
```

> 配置文件位置：
> - macOS：`~/Library/Application Support/Claude/claude_desktop_config.json`
> - Windows：`%APPDATA%\Claude\claude_desktop_config.json`

## Cursor / Cline Integration

Cursor 与 Cline 均支持 MCP，在其 MCP 配置面板加入相同结构：

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

## Streamable HTTP（远程部署）

```bash
CSM_TRANSPORT=streamable-http china-stock-mcp
```

客户端按其文档接入 streamable-http 端点。

---

## Development

```bash
# 同步依赖（含全部 optional extras 与 dev 组）
uv sync --all-extras --group dev

# Lint
uv run ruff check src/ tests/

# Type check（strict 模式）
uv run mypy --strict src/china_stock_mcp/

# 运行测试 + 核心覆盖率（≥ 70% 阈值）
uv run pytest tests/ \
  --cov=src/china_stock_mcp \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=70
```

`pytest-asyncio` 已配置为 `auto` 模式；属性测试基于 [`hypothesis`](https://hypothesis.works/)，对应 design 文档中编号 P1–P18 的 correctness properties。
覆盖率门槛聚焦核心业务逻辑；外部数据源 adapter 与 FastMCP 装配入口依赖第三方网络 / 协议边界，已在 coverage 配置中排除。

CI 工作流见 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)，在 Python 3.11 与 3.12 矩阵下跑 ruff + mypy + pytest + coverage。

---

## Project Layout

```
src/china_stock_mcp/
├── server.py              # FastMCP 入口，注册 tools / prompts / resources
├── config.py              # 环境变量 → Settings
├── exceptions.py          # ChinaStockMCPError 异常树
├── models.py              # Pydantic v2 DTO
├── normalizer.py          # 代码归一化
├── cache.py               # TTL 分级缓存（diskcache / redis）
├── rate_limiter.py        # Token Bucket 全局限流
├── formatters.py          # Markdown 渲染 + 免责声明
├── adapters/              # akshare / tushare / efinance + base
├── services/              # 业务编排层
├── tools/                 # 10 个 MCP tool
├── prompts/               # 3 个 MCP prompt
└── resources/             # 3 个 MCP resource
```

---

## License

[MIT](LICENSE)

---

## Disclaimer

> ⚠️ 数据来源于公开第三方，可能存在延迟或误差。本服务仅供研究学习使用，不构成任何投资建议。

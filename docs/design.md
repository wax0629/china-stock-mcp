# China Stock MCP - 详细设计文档

> 版本：v1.0  
> 状态：设计稿  
> 目的：为后续进入 Spec 模式实现做依据

---

## 1. 项目定位

### 1.1 一句话定义
**China Stock MCP** 是一个面向中文金融市场（A股 / 港股 / 公募基金）的 Model Context Protocol 服务器，让任何支持 MCP 的 AI 客户端（Claude Desktop、Cursor、Cline 等）即时获得"懂中国市场的分析师"能力。

### 1.2 项目目标
- **首要目标**：成为中文金融领域事实上的 MCP 标准接入层之一
- **次要目标**：本地隐私优先，数据不出用户机器
- **非目标（明确排除）**：
  - 不做实时 L1/L2 行情（合规与成本问题）
  - 不做下单交易（首版本不涉及，避免责任风险）
  - 不做预测/荐股（仅提供数据与计算，不输出"买/卖"建议）
  - 不做前端 UI（UI 由 MCP 客户端承担）

### 1.3 目标用户
1. **个人投资者**：希望用 AI 辅助看盘、做基本面研究的散户
2. **量化爱好者**：需要快速验证想法、获取结构化数据
3. **AI 应用开发者**：基于该 MCP 构建上层金融 Agent
4. **财经内容创作者**：需要数据驱动写作

---

## 2. 总体架构

### 2.1 分层结构

```
┌────────────────────────────────────────────────┐
│  MCP Client (Claude Desktop / Cursor / Cline)  │
└──────────────────────┬─────────────────────────┘
                       │ MCP Protocol (stdio / streamable-http)
┌──────────────────────▼─────────────────────────┐
│              FastMCP Server Layer              │
│  ┌──────────┬──────────┬──────────┬─────────┐  │
│  │  tools   │ prompts  │resources │ context │  │
│  └──────────┴──────────┴──────────┴─────────┘  │
├────────────────────────────────────────────────┤
│              Service Layer                     │
│  ┌────────────────────────────────────────┐   │
│  │  Quote / Fundamental / Fund / Screen   │   │
│  │  Money Flow / Industry / Search        │   │
│  └────────────────────────────────────────┘   │
├────────────────────────────────────────────────┤
│            Data Adapter Layer                  │
│  ┌──────────┬──────────┬──────────┬─────────┐  │
│  │ akshare  │ tushare  │ efinance │ fallback│  │
│  └──────────┴──────────┴──────────┴─────────┘  │
├────────────────────────────────────────────────┤
│       Cache Layer (diskcache + TTL)            │
├────────────────────────────────────────────────┤
│       Utils (symbol normalize / format / log)  │
└────────────────────────────────────────────────┘
```

### 2.2 模块职责

| 模块 | 职责 | 关键约束 |
|------|------|----------|
| **FastMCP Layer** | 协议适配、工具注册、参数校验 | 不写业务逻辑 |
| **Service Layer** | 业务编排、多源融合、指标计算 | 不直接接触协议 |
| **Data Adapter** | 封装第三方库，提供统一接口 | 屏蔽数据源差异 |
| **Cache Layer** | TTL 缓存，减少外部调用 | 不同数据用不同 TTL |
| **Utils** | 代码归一化、Markdown 渲染、日志 | 纯函数优先 |

### 2.3 目录结构

```
china-stock-mcp/
├── pyproject.toml
├── README.md
├── LICENSE                          # MIT
├── .python-version
├── src/
│   └── china_stock_mcp/
│       ├── __init__.py
│       ├── server.py                # FastMCP 入口
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── search.py
│       │   ├── quote.py
│       │   ├── kline.py
│       │   ├── fundamental.py
│       │   ├── financial.py
│       │   ├── money_flow.py
│       │   ├── industry.py
│       │   ├── fund.py
│       │   └── screen.py
│       ├── prompts/
│       │   ├── __init__.py
│       │   ├── research_report.py
│       │   └── valuation_compare.py
│       ├── resources/
│       │   ├── __init__.py
│       │   └── market_overview.py
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── akshare_adapter.py
│       │   ├── tushare_adapter.py
│       │   └── base.py              # 抽象基类
│       ├── services/
│       │   ├── __init__.py
│       │   ├── symbol_service.py
│       │   ├── quote_service.py
│       │   └── ...
│       ├── cache.py
│       ├── formatters.py            # Markdown 渲染
│       ├── normalizer.py            # 代码归一化
│       ├── exceptions.py
│       └── config.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── docs/
    ├── tools.md                     # 工具完整列表
    ├── examples.md                  # 使用示例
    └── deployment.md
```

---

## 3. 工具（Tools）设计

### 3.1 设计原则

1. **AI-First**：参数和返回都按"AI 易用"设计，而不是"程序员易用"
2. **模糊容错**：所有 symbol 参数同时接受中文名、6 位代码、带后缀代码
3. **Markdown 输出**：默认返回 Markdown 表格 / 段落，AI 可直接转述
4. **自带解读上下文**：返回值附带"该指标处于行业 X 分位"等参考信息
5. **错误信息对 AI 友好**：返回结构化错误说明，不抛 traceback
6. **Token 控制**：单次返回控制在 ~3000 token 以内

### 3.2 工具清单（v1.0 - 10 个核心工具）

#### 3.2.1 `search_symbol`
**用途**：模糊搜索股票/基金代码  
**签名**：
```python
def search_symbol(query: str, market: str = "all") -> str
```
**参数**：
- `query`：中文名、拼音首字母、代码片段（必填）
- `market`：`a_stock` / `hk_stock` / `fund` / `all`（默认 all）

**返回**：Markdown 表格，最多 5 个匹配
```
| 代码      | 名称       | 市场  | 行业       |
|-----------|------------|-------|------------|
| 300750.SZ | 宁德时代   | A股   | 电池        |
```

#### 3.2.2 `get_quote`
**用途**：获取实时行情快照  
**签名**：
```python
def get_quote(symbol: str | list[str]) -> str
```
**返回内容**：现价、涨跌幅、涨跌额、成交量、成交额、换手率、市盈率（动）、市净率、总市值、流通市值

**特殊设计**：
- 接受 symbol 列表，最多 20 个，批量返回对比表
- 数据时延：标注"数据延迟约 15 分钟"

#### 3.2.3 `get_kline`
**用途**：获取 K 线 + 技术指标  
**签名**：
```python
def get_kline(
    symbol: str,
    period: str = "daily",        # daily / weekly / monthly / 60min / 30min
    count: int = 60,              # 最近 N 根，最大 250
    adjust: str = "qfq",          # qfq 前复权 / hfq 后复权 / none 不复权
    indicators: list[str] = ["MA20", "MA60", "MACD"]
) -> str
```
**返回**：K 线表格 + 自动计算的指标列 + 形态简评（如"近 5 日呈缩量回调"）

#### 3.2.4 `get_fundamentals`
**用途**：基本面快照  
**签名**：
```python
def get_fundamentals(symbol: str) -> str
```
**返回内容**：
- 估值：PE-TTM、PE-动态、PB、PS、PEG
- 盈利：ROE、ROA、毛利率、净利率
- 成长：营收增速、净利润增速（同比/环比）
- 财务健康：资产负债率、流动比率、经营现金流/净利润
- **每个指标自动附行业分位数**

#### 3.2.5 `get_financial_report`
**用途**：财务三表关键科目  
**签名**：
```python
def get_financial_report(
    symbol: str,
    report_type: str = "annual",   # annual / quarterly
    periods: int = 4               # 最近 N 期
) -> str
```
**返回**：营收、归母净利、扣非净利、毛利、经营现金流、总资产、负债、净资产 等关键科目的多期对比

#### 3.2.6 `get_money_flow`
**用途**：资金流向（这是中文市场独有差异点）  
**签名**：
```python
def get_money_flow(
    symbol: str = None,            # 不传则返回全市场榜
    flow_type: str = "north",      # north / main / dragon_tiger
    top_n: int = 20
) -> str
```
**类型**：
- `north`：北向资金（沪股通+深股通）
- `main`：主力资金净流入
- `dragon_tiger`：龙虎榜

#### 3.2.7 `get_industry_peers`
**用途**：同行业对比  
**签名**：
```python
def get_industry_peers(
    symbol: str,
    metrics: list[str] = ["pe", "pb", "roe", "revenue_growth"],
    top_n: int = 10
) -> str
```
**返回**：以该股为基准，同行业按选定指标排名的对比表

#### 3.2.8 `get_fund_info`
**用途**：基金综合信息  
**签名**：
```python
def get_fund_info(fund_code: str) -> str
```
**返回**：基金经理、规模、成立日期、近 1/3/6/12 月收益、最大回撤、夏普、同类排名、重仓股 Top10、行业分布

#### 3.2.9 `screen_stocks`
**用途**：选股器  
**签名**：
```python
def screen_stocks(
    criteria: dict,                # 见下
    sort_by: str = "market_cap",
    order: str = "desc",
    limit: int = 30
) -> str
```
**criteria 示例**：
```python
{
    "pe_ttm": {"min": 0, "max": 20},
    "roe": {"min": 15},
    "market_cap": {"min": 5e10},   # 单位：元
    "industry": ["新能源车", "电池"]
}
```

#### 3.2.10 `get_market_overview`
**用途**：市场总览  
**签名**：
```python
def get_market_overview() -> str
```
**返回**：上证/深证/创业板指数、涨跌家数、涨停跌停数、北向净流入、主力净流入行业 Top5、市场热度评分

### 3.3 参数校验

- 所有工具入口使用 Pydantic 模型校验
- 失败返回结构化错误：
```
错误：参数 'symbol' 无法识别 '苹果'。
- 该 MCP 仅支持 A 股 / 港股 / 公募基金
- 如需查询美股，请使用其他 MCP 服务
- 您是否想查询 '苹果园(831175)' 或 '苹果手机概念股'？
```

---

## 4. Prompts 设计（v1.0）

Prompt 是 MCP 协议的一等公民，AI 会主动加载这些"模板"。

### 4.1 `research_report`
**用途**：生成完整投研报告  
**参数**：`symbol`、`report_length`（short/standard/deep）  
**内置流程**：调用 fundamentals → financial_report → industry_peers → money_flow → kline → 综合输出

### 4.2 `valuation_compare`
**用途**：同业估值对比  
**参数**：`symbols`（list）

### 4.3 `weekly_review`
**用途**：自选股周复盘  
**参数**：`symbols`、`week`

---

## 5. Resources 设计（v1.0）

### 5.1 `market://overview/today`
- 当日市场总览快照，每 5 分钟更新
- AI 客户端可"订阅"

### 5.2 `market://north-flow/latest`
- 最新北向资金分布

### 5.3 `symbol://{code}/profile`
- 单个标的的静态资料（公司简介、主营业务、上市日期等）

---

## 6. 数据源策略

### 6.1 主备源选择

| 数据类型 | 主源 | 备源 | TTL |
|----------|------|------|-----|
| 实时行情 | akshare (东财) | efinance | 60s |
| K 线 | akshare | tushare | 5min |
| 财务报表 | akshare | tushare | 24h |
| 北向资金 | akshare | - | 5min |
| 龙虎榜 | akshare | - | 24h |
| 基金净值 | akshare | efinance | 1h |
| 公司信息 | akshare | - | 7d |

### 6.2 容错策略

```python
async def fetch_with_fallback(primary, fallback, *args):
    try:
        return await primary(*args)
    except (NetworkError, RateLimitError) as e:
        logger.warning(f"primary failed: {e}, falling back")
        return await fallback(*args)
    except DataNotFoundError:
        raise  # 数据确实不存在，不切换
```

### 6.3 限流保护

- 全局 token bucket：默认 30 req/min（保守）
- 每个数据源单独配额
- 超限时返回："数据源调用频率过高，请稍后重试，或调高 ttl 设置"

---

## 7. 缓存策略

### 7.1 选型
- **diskcache**（默认）：零依赖、本地落盘、适合个人使用
- **redis**（可选）：通过环境变量切换，适合多用户/多实例部署

### 7.2 缓存键设计

```
{tool_name}:{normalized_symbol}:{param_hash}:v{schema_version}
```

`schema_version` 用于发布新版本时强制失效旧缓存。

### 7.3 TTL 分级

| 级别 | 时间 | 适用 |
|------|------|------|
| HOT | 60s | 实时行情 |
| WARM | 5min | K线、资金流 |
| COLD | 1h | 基金净值、市场总览 |
| FROZEN | 24h | 财务数据 |
| STATIC | 7d | 公司信息、行业分类 |

---

## 8. 关键工具内部规范

### 8.1 代码归一化（normalizer）

输入 → 标准化输出（带后缀）：

| 输入 | 输出 |
|------|------|
| `300750` | `300750.SZ` |
| `600519` | `600519.SH` |
| `00700` | `00700.HK` |
| `宁德时代` | `300750.SZ` |
| `300750.SZ` | `300750.SZ`（保持） |
| `Ningde` / 拼音 | 通过映射表查找 |

后缀规则：
- 60xxxx / 68xxxx / 90xxxx → `.SH`
- 00xxxx / 30xxxx / 20xxxx → `.SZ`
- 8xxxxx → `.BJ`（北交所）
- 港股 5 位数字 → `.HK`

### 8.2 Markdown 渲染规范

- 表格：使用 `|--|--|`，列对齐右对齐数字、左对齐文本
- 数字格式：金额自动转"亿"/"万"，百分比保留 2 位小数
- 涨跌：用 `🔴` 红涨 / `🟢` 绿跌（A 股习惯，与国外相反！）
- 提示信息：用 blockquote `>` 或斜体 `*`

### 8.3 错误处理

定义统一异常树：
```
ChinaStockMCPError
├── SymbolError          # 代码识别失败
├── DataSourceError      # 数据源问题
│   ├── RateLimitError
│   └── NetworkError
├── DataNotFoundError    # 数据确实不存在
└── ValidationError      # 参数错误
```

每个异常都有面向 AI 的 `to_user_message()` 方法。

---

## 9. 配置与部署

### 9.1 配置项（环境变量）

| 变量 | 默认 | 说明 |
|------|------|------|
| `CSM_CACHE_BACKEND` | `disk` | `disk` / `redis` |
| `CSM_CACHE_DIR` | `~/.cache/china-stock-mcp` | 缓存目录 |
| `CSM_LOG_LEVEL` | `INFO` | 日志级别 |
| `CSM_TUSHARE_TOKEN` | None | tushare token，可选 |
| `CSM_RATE_LIMIT` | `30` | 每分钟最大请求 |
| `CSM_DATA_DELAY_NOTICE` | `true` | 是否在返回中加延迟提示 |

### 9.2 传输模式

- **stdio**（默认）：本地 Claude Desktop / Cursor 集成
- **streamable-http**：远程部署，多用户共享

### 9.3 安装方式

```bash
# 方式 1：uvx 一键运行（推荐）
uvx china-stock-mcp

# 方式 2：本地开发
git clone https://github.com/yourname/china-stock-mcp
cd china-stock-mcp
uv sync
uv run python -m china_stock_mcp.server
```

### 9.4 客户端配置示例

**Claude Desktop** (`claude_desktop_config.json`)：
```json
{
  "mcpServers": {
    "china-stock": {
      "command": "uvx",
      "args": ["china-stock-mcp"],
      "env": {
        "CSM_CACHE_DIR": "C:\\Users\\yourname\\.cache\\csm"
      }
    }
  }
}
```

---

## 10. 测试策略

### 10.1 测试金字塔

- **单元测试**（多）：normalizer、formatters、cache、各 service 的纯逻辑
- **集成测试**（中）：mock 数据源，验证 tool → service → adapter 链路
- **契约测试**（少）：定期跑真实数据源，验证 schema 没变（CI 跑）
- **MCP 协议测试**：用 `fastmcp dev` 命令交互式验证每个 tool 注册正确

### 10.2 关键测试用例

- 模糊代码识别准确率
- 数据源切换的容错链
- 缓存 TTL 行为
- 错误信息的可读性（人工 review）
- 单次返回 token 数（pytest 中断言上限）

### 10.3 数据 fixtures

- 把每个数据源的真实返回保存为 JSON fixture
- 单元测试用 fixture，不打真实网络

---

## 11. 安全与合规

### 11.1 免责声明
每次工具返回末尾自动追加：
> ⚠️ 数据来源于公开第三方，可能存在延迟或误差。本服务仅供研究学习使用，不构成任何投资建议。

### 11.2 敏感信息
- 不收集用户身份/持仓/交易记录
- 日志默认不记录用户输入的具体 query
- Tushare token 等密钥仅从环境变量读取，不落盘

### 11.3 速率与公平
- 全局限流防止滥用免费数据源
- 优先使用缓存，对数据源友好

---

## 12. 版本与发布

### 12.1 v1.0 验收标准（DoD）

- [ ] 10 个核心工具全部实现
- [ ] 单元测试覆盖率 ≥ 70%
- [ ] 在 Claude Desktop 上端到端跑通 5 个典型场景
- [ ] README 包含安装、配置、3 个示例 GIF
- [ ] 提交至 awesome-mcp-servers 列表
- [ ] PyPI 发布
- [ ] 文档站（用 Mintlify 或 mdBook）

### 12.2 语义化版本
- 0.x.x：内部预发，工具 schema 可能 breaking
- 1.0.0：稳定首版
- 1.x.x：新增工具/字段（向后兼容）
- 2.0.0：协议或 schema 大改

---

## 13. 风险与对策

| 风险 | 等级 | 对策 |
|------|------|------|
| akshare 接口变更 | 高 | 适配层隔离 + 契约测试 + 备用源 |
| 数据源限流封禁 | 中 | 缓存 + 限流 + 多源切换 |
| 用户拿来做交易决策出问题 | 高 | 强免责 + 文档警示 + 不做"建议"输出 |
| MCP 协议演进 | 中 | 紧跟 FastMCP 主版本 |
| 中文金融 MCP 竞争者出现 | 中 | 用速度和深度先占位 |

---

## 14. 开发优先级（给 Spec 模式参考）

按以下顺序拆 Task，每个 Task 力求 1-2 小时可完成：

**Phase 1：地基**
1. 项目初始化（pyproject、uv、目录骨架）
2. normalizer 实现 + 测试
3. cache 模块实现 + 测试
4. exceptions 模块
5. formatters 模块（Markdown 渲染）
6. base adapter 抽象

**Phase 2：第一条链路打通**
7. akshare adapter 实现 search、quote
8. tools/search.py 实现
9. tools/quote.py 实现
10. server.py 装配 + Claude Desktop 端到端联调

**Phase 3：核心工具补齐**
11. tools/kline.py
12. tools/fundamental.py
13. tools/financial.py
14. tools/money_flow.py
15. tools/industry.py
16. tools/fund.py
17. tools/screen.py
18. tools/market_overview.py（兼 resource）

**Phase 4：增强**
19. prompts 模块
20. resources 模块
21. tushare 备用 adapter
22. 限流模块

**Phase 5：发布**
23. 完整 README + GIF
24. CI / 测试覆盖
25. PyPI 发布
26. 提交 awesome 列表

---

## 附录 A：示例对话

**用户**：宁德时代和比亚迪现在哪个更值得关注？

**AI 内部调用**：
1. `search_symbol("宁德时代")` → 300750.SZ
2. `search_symbol("比亚迪")` → 002594.SZ
3. `get_quote(["300750.SZ", "002594.SZ"])` → 行情对比
4. `get_fundamentals("300750.SZ")` + `get_fundamentals("002594.SZ")` → 基本面
5. `get_money_flow(symbol="300750.SZ", flow_type="north")` + 同上比亚迪
6. 综合输出对比报告

**用户**：帮我筛选新能源车板块里 PE 小于 30、ROE 大于 15 的股票

**AI 内部调用**：
1. `screen_stocks({"industry": ["新能源车"], "pe_ttm": {"max": 30}, "roe": {"min": 15}})` → 名单

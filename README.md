# gautils

通用 Python 工具库。旧 API 保持原导入路径；新代码建议使用严格、可版本化的
`gautils.v2` API。

## 安装

```bash
pip install -e .
pip install -r requirements-dev.txt  # 开发与测试
```

## v2 HTTP

v2 复用连接池，默认设置连接/读取超时，并对 HTTP 错误抛出
`HttpRequestError`。调用者始终得到 `requests.Response`，不再混用字符串、响应对象
和 `None`。

```python
from gautils.v2 import HttpClient, RetryPolicy, Timeout

with HttpClient(
    timeout=Timeout(connect=5, read=30),
    retry=RetryPolicy(total=3),
) as client:
    response = client.get("https://example.com/api")
    data = response.json()
```

## v2 Database

数据库 API 统一使用 SQLAlchemy Core、命名参数和显式事务。MySQL URL 通过
`URL.create()` 构造，密码和查询参数不再手工拼接。

```python
from gautils.v2 import Database

with Database.mysql(
    user="user",
    password="secret",
    host="127.0.0.1",
    database="app",
    query={"charset": "utf8mb4"},
) as db:
    users = db.fetch_all(
        "SELECT * FROM users WHERE status=:status",
        {"status": "active"},
    )
```

批量写入返回结构化 `BatchResult`：

```python
result = db.upsert_dataframe(
    "users", frame, key_columns=["id"], batch_size=1000,
)
result.raise_for_errors()
```

## v2 Feishu

```python
from gautils.v2 import Feishu

feishu = Feishu(app_id="app_id", app_secret="app_secret")
table = feishu.bitable("app_token").table(name="数据表")
records = table.records()
result = table.upsert(frame)
```

当前锁定的 `lark-oapi==1.4.24` 没有电子表格单元格读写 API。v2 对这些操作明确
抛出 `UnsupportedOperationError`，不会伪装成空数据或成功结果。

## 旧版兼容

以下旧入口继续保留：

```python
from gautils import Web, connect_mysql
from gautils.feishu import Feishu
```

旧 API 的返回类型与主要调用方式不变。确定的实现缺陷会修复，但新增的严格错误语义
只在 `gautils.v2` 中启用。

## 验证

```bash
python -m pytest -q
```

项目门禁要求 `gautils` 生产代码语句覆盖率不低于 98%。

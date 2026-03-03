# gautils 项目 AI 编码规范

## 项目结构

```
gautils/
├── gautils/              # 主代码包
│   ├── feishu/           # 飞书相关功能
│   │   ├── core/         # 核心实现
│   │   │   ├── bitable.py
│   │   │   ├── spreadsheets.py
│   │   │   └── __init__.py
│   │   ├── tools.py
│   │   └── __init__.py
│   ├── mysqldb.py        # MySQL 数据库操作
│   ├── web.py            # Web 请求工具
│   ├── utils.py          # 通用工具函数
│   └── __init__.py
├── test_gautils/         # 单元测试
│   ├── test_bitable.py
│   ├── test_spreadsheet.py
│   └── __init__.py
├── docs/                 # 文档
├── temp/                 # 临时文件
└── .qoder/               # AI 配置
```

## 编码风格

### 1. 导入规范

```python
# 标准库
import warnings
import json

# 第三方库
import pandas as pd
import lark_oapi as lark

# 项目内部导入（使用相对导入）
from ...utils import batch_split
from ..conf import global_conf
```

### 2. 类命名规范

- 使用 PascalCase
- 内部类使用 `_` 前缀或嵌套类
- 常量类使用大写下划线

```python
class _FS:
    """内部常量类"""
    class BITABLE:
        class TABLE:
            ID = 'id'
            NAME = 'name'
            class FIELD:
                V_TEXT = 1
                V_NUMBER = 2

class BiTable:
    """飞书多维表格"""
    pass

class TableField:
    """表格字段定义"""
    class FieldType(Enum):
        Text = 1
        Number = 2
```

### 3. 方法命名规范

- 使用 snake_case
- 私有方法使用 `_` 前缀
- 查询方法使用 `query_` 前缀
- 获取单个对象使用 `get_` 前缀

```python
def query_sheets(self) -> List[dict]:
    """查询所有工作表"""
    pass

def get_sheet(self, sheet_id: str = None) -> Optional[Sheet]:
    """获取单个工作表"""
    pass

def _build_range(self, start_cell: str) -> str:
    """内部辅助方法"""
    pass
```

### 4. 文档字符串规范

使用三重双引号，包含功能描述和参数说明：

```python
def create_sheet(self, title: str, row_count: int = 1000) -> Optional[Sheet]:
    """创建工作表

    Args:
        title: 工作表标题
        row_count: 行数，默认1000

    Returns:
        新创建的Sheet对象，失败返回None
    """
    pass
```

### 5. 错误处理规范

- 使用 lark.logger 记录错误
- 返回 None 或 False 表示失败
- 不抛出异常（除非严重错误）

```python
if not response.success():
    lark.logger.error(
        f"api_call failed, code: {response.code}, "
        f"msg: {response.msg}, log_id: {response.get_log_id()}"
    )
    return None
```

### 6. 类型注解规范

- 使用 Python 3.9+ 类型注解
- 可选类型使用 `Optional[T]`
- 列表类型使用 `List[T]`
- 返回值类型必须标注

```python
from typing import Optional, List

def query_sheets(self) -> List[dict]:
    pass

def get_sheet(self, sheet_id: str = None) -> Optional[Sheet]:
    pass
```

### 7. 飞书 API 调用规范

- 使用类型注解标注 request/response
- 使用 builder 模式构建请求
- 处理 has_more 分页（如需要）

```python
from lark_oapi.api.sheets.v3 import QuerySpreadsheetSheetRequest, QuerySpreadsheetSheetResponse

request: QuerySpreadsheetSheetRequest = QuerySpreadsheetSheetRequest.builder() \
    .spreadsheet_token(self.spreadsheet_token) \
    .build()

response: QuerySpreadsheetSheetResponse = self.client.sheets.v3.spreadsheet_sheet.query(request)

if not response.success():
    lark.logger.error(f"...")
    return []
```

### 8. 单元测试规范

- 测试类继承 `unittest.TestCase`
- 类名使用 `Test` 前缀
- 方法名使用 `test_` 前缀
- 使用 `setUpClass` 进行初始化

```python
import unittest
from gautils.feishu.core import Spreadsheet, Feishu

class TestSpreadsheet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fs = Feishu('app_id', 'app_secret')
        cls.spreadsheet = cls.fs.get_spreadsheet('token')

    def test_query_sheets(self):
        sheets = self.spreadsheet.query_sheets()
        self.assertIsInstance(sheets, list)
```

### 9. 缓存处理规范

- 使用 `_` 前缀标记私有缓存变量
- 在修改操作后清除缓存

```python
def __init__(self):
    self._sheets = None  # 缓存

def query_sheets(self):
    if self._sheets is not None:
        return self._sheets
    # ... 查询逻辑
    self._sheets = sheets
    return sheets

def create_sheet(self, title):
    # ... 创建逻辑
    self._sheets = None  # 清除缓存
```

### 10. DataFrame 处理规范

- 使用 pandas 处理数据
- 空值处理使用 `pd.DataFrame()` 而非 `None`
- 列名使用字符串索引

```python
import pandas as pd

def read(self) -> pd.DataFrame:
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    df.columns = [self._col_index_to_letter(i) for i in range(max_cols)]
    return df
```

### 11. 注释规范

- 如无必要不要写注释。只有容易误解或者逻辑非常复杂时简要解释
- 写也要尽可能简短

## 文件操作规范

1. **读取文件**：使用 `utf-8` 编码
2. **路径处理**：使用原始字符串 `r'path'`
3. **配置文件**：使用 YAML 格式

## 版本控制规范

1. **Git 提交信息**：使用中文，不超过 50 字
2. **测试先行**：修改代码前确保有对应测试
3. **不修改外部文件**：只修改本目录内文件

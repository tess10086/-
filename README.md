# 信用卡客户下月违约风险分析系统

> 计算机金融学导论 · 课程综合项目 · **题目一**
> 识别潜在违约客户，辅助安排风险核查。
> ⚠️ 本项目仅作教学分析，**不用于真实授信决策**。

---

## 一、快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动界面
streamlit run app.py
```

启动后浏览器会自动打开 `http://localhost:8501`。
数据文件 `01_credit_default.csv` 已随仓库提供，**无需额外下载**。

---

## 二、界面结构（四个页面）

| 页面 | 内容 |
|---|---|
| **数据概览** | 统计卡片（三组样本量/违约率）、按划分统计表、授信额度分布、年龄分布、还款状态分布、额度使用率分箱 vs 违约率、还款状态编码含义 |
| **模型比较** | C 同学交付的验证集模型对比表、阈值—指标曲线、特征重要性 Top20；下方为界面现场训练的交互式选型与阈值选择 |
| **最终评价（测试集）** | D 同学锁定的测试集最终结果（读文件不重跑）、混淆矩阵、指标卡；附在线复算核对 |
| **客户风险演示** | 选一条客户 → 风险概率、所用阈值、风险提示、主要输入信息；测试集记录会单独框出事后真实标签 |

---

## 三、数据与方法（严格遵循课程要求）

- **数据**：`01_credit_default.csv`，30,000 条客户历史记录；train / valid / test = 20,999 / 4,500 / 4,501 条。
- **标签**：`default_next_month`，1 表示下月违约。
- **划分**：分层随机划分（数据无逐笔申请日期）。
- **预处理**：数字列缺失填补 + 标准化；类别列字符串化 + 缺失填补 + 独热编码；
  还款状态列**先转字符串再独热编码，负值 -2 / -1 是有效编码，不作为缺失处理**。
  **预处理仅在训练集上拟合**。
- **模型**：全判不违约基线、逻辑回归、随机森林（`n_estimators=300, max_depth=12,
  min_samples_leaf=5, class_weight='balanced'`）。
- **阈值**：仅用**验证集**确定，最终选定 **0.5**（Recall≥0.6 约束下 Precision 最高）。
- **测试集**：只评价一次。

---

## 四、目录结构

```
.
├─ app.py                     # 界面层（四个页面）
├─ credit_pipeline.py         # 计算逻辑：数据核查 / 预处理 / 训练 / 阈值 / 评价
├─ artifacts_io.py            # 队友交付物接入层（只读不重算）
├─ preprocess.py              # 特征工程模块（被 best_model.pkl 反序列化时引用）
├─ smoke_test.py              # 四页冒烟测试
├─ requirements.txt
├─ 01_credit_default.csv
└─ artifacts/                 # 各成员交付物
   ├─ best_model.pkl                 (C) 锁定的随机森林模型
   ├─ threshold.json                 (C) 选定阈值与选择依据
   ├─ model_comparison.csv           (C) 三模型 train/valid 对比
   ├─ threshold_comparison.csv       (C) 验证集阈值扫描表
   ├─ feature_importance.csv         (C) 特征重要性排名
   ├─ test_result.json               (D) 测试集最终评价（只跑一次，锁定）
   └─ 字段口径.md                     (B) 字段表与编码说明
```

---

## 五、最终测试集结果（锁定，只评价一次）

| 指标 | 值 |
|---|---|
| ROC-AUC | **0.782545** |
| 违约类精确率 | 0.495840 |
| 召回率 | 0.598394 |
| 混淆矩阵 (TN/FP/FN/TP) | 2899 / 606 / 400 / 596 |
| 测试集样本量 | 4,501（违约 996 条，违约率 0.2213） |

结果来源：`artifacts/test_result.json`（由 D 同学在测试集上只运行一次后锁定）。
界面「最终评价」页读取该文件，**不重新计算**。
用 C 同学的 `best_model.pkl` 在线复算，结果与上述逐项一致。

---

## 六、成员分工

| 角色 | 姓名 | 负责内容 | 交付文件 |
|---|---|---|---|
| A | 刘思雨 | 数据核查 + 数据概览页 | `src/data_io.py`、`src/data_check.py` |
| B | 赵梓伶 | 特征预处理 + 防泄漏 | `src/preprocess.py`、`docs/字段口径.md` |
| C | 冯荟如 | 模型训练 + 阈值选择 | `src/models.py`、`artifacts/*.pkl/csv/json` |
| D | 吴姿仪 | 测试集评价（只跑一次） | `src/evaluate.py`、`artifacts/test_result.json` |
| E | 陈韬伊 | 界面 + 演示 + PPT（组长） | `app.py`、`artifacts_io.py`、`README.md`、`slides/` |

### E（陈韬伊）本人实现说明

界面层**不参与任何建模**：C 的模型、D 的测试结果、B 的字段口径全部以文件形式接入，
页面只读取与展示，符合"测试集只评价一次""页面上的数必须由代码算出"的红线。

- `artifacts_io.py`：统一读取 `artifacts/` 下的交付物，含加载状态与错误提示。
  关键函数：`load_c_bundle` / `predict_with_c_model` / `load_d_test_result` /
  `load_c_model_comparison` / `load_c_threshold_curve` / `load_c_feature_importance`。
- `app.py`：四个可切换页面 + 加载状态 + 读文件失败提示 + 无效 record_id 提示。
- `smoke_test.py`：基于 `streamlit.testing.v1.AppTest` 的四页冒烟测试。

---

## 七、测试记录

```bash
python smoke_test.py
```

```
[数据概览]        exception=0
[模型比较]        exception=0
[最终评价（测试集）] exception=0
[客户风险演示]     exception=0
```

另做了一次一致性核验：用 C 同学的 `best_model.pkl` 在测试集复算，
得到 ROC-AUC 0.782545、精确率 0.495840、召回率 0.598394、
混淆矩阵 2899/606/400/596，与 D 同学的 `artifacts/test_result.json` **逐项完全一致**。

---

## 八、免责与说明

- 本项目**仅作教学分析**，不构成授信决定，不得用于真实风险核查。
- 模型使用了性别、婚姻、教育、年龄等**敏感属性**；**相关性不等于因果关系**，
  输出不得作为对任何个人或群体差别对待的依据。
- 数据无逐笔申请日期，划分为分层随机划分，结论不可外推到未来时段。
- 数据来源：UCI *Default of Credit Card Clients* 数据集的课程改写版，`record_id` 仅用于追溯。

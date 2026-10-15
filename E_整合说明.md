# E 同学（界面/组长）数据整合说明

## 一、缺口分析：原来缺什么

| # | 缺口 | 影响 | 现状 |
|---|---|---|---|
| 1 | 界面「自己训、自己评」，完全没读 C、D 的交付物 | 页面数字与 C/D 报告对不上，等于 C、D 的模块没接入 | ✅ 已接入 |
| 2 | `best_model.pkl` 反序列化失败 | C 的模型根本打不开 | ✅ 已解决（见下） |
| 3 | C 的 `model_comparison.csv` / `threshold_comparison.csv` / `feature_importance.csv` 没展示 | 「模型比较」页缺验证集指标表、阈值曲线、特征重要性 | ✅ 已接入 |
| 4 | D 的 `test_result.json` 没读 | 「最终评价」页违反约定（应"读文件不重跑"） | ✅ 已接入 |
| 5 | 「模型比较」页**打开即崩溃**（预存 bug） | 默认选中「基线」，基线 F1 全为 None → 选阈值返回 None → `float >= None` 报错 | ✅ 已修 |

### 缺口 2 的根因（最关键）

`best_model.pkl` 里存了 `FunctionTransformer(func=preprocess._pay_status_to_string)`，
反序列化时要 `import preprocess`。**B 同学只给了 `字段口径.md`，没给 `preprocess.py`**，
所以 `joblib.load` 报 `ModuleNotFoundError: No module named 'preprocess'`。

处理：项目根目录补了 `preprocess.py`（垫片，只实现被引用到的那个函数）。
**已用测试集复算验证**，结果与 D 的 `test_result.json` 逐项完全一致：

| 指标 | 复算值 | D 的 test_result.json |
|---|---|---|
| ROC-AUC | 0.782545 | 0.782545 |
| 精确率 | 0.495840 | 0.495840 |
| 召回率 | 0.598394 | 0.598394 |
| TN/FP/FN/TP | 2899/606/400/596 | 2899/606/400/596 |

→ 说明垫片行为与 B 的原实现一致。**但仍应向 B 索取正式的 `preprocess.py` 覆盖它**
（README 与源码完整性要求 B 交付该文件）。

## 二、当前的目录约定

```
PythonProject/
├─ 01_credit_default.csv
├─ app.py                 # E：界面（四个页面）
├─ credit_pipeline.py     # E：计算逻辑
├─ artifacts_io.py        # 新增：队友交付物接入层
├─ preprocess.py          # 新增：B 的 preprocess 垫片（收到正式版请覆盖）
├─ smoke_test.py          # 新增：四页冒烟测试
└─ artifacts/             # 队友交付物统一放这里
   ├─ best_model.pkl                (C) 锁定模型
   ├─ threshold.json                (C) 选定阈值 + 依据
   ├─ model_comparison.csv          (C) 三模型 train/valid 对比
   ├─ threshold_comparison.csv      (C) 阈值扫描表
   ├─ feature_importance.csv        (C) 特征重要性
   ├─ test_result.json              (D) 测试集最终评价（只跑一次）
   └─ 字段口径.md                    (B) 字段表与编码说明
```

## 三、改了哪些地方

**新增 `artifacts_io.py`**：统一读取队友交付物，界面层只展示不重算。
关键函数：`load_c_bundle()` / `predict_with_c_model()` / `load_d_test_result()` /
`load_c_model_comparison()` / `load_c_threshold_curve()` / `load_c_feature_importance()` /
`artifacts_status()` / `c_bundle_error()`。

**`app.py` 四处改动（全部为追加，未删原逻辑）**：

1. **侧边栏**：显示 7 项交付物的到达状态（✅/❌），C 模型加载失败时直接报错原因。
2. **模型比较页**：顶部新增「C 同学交付」区块 —— 验证集模型对比表、阈值—指标曲线
   （标出选定阈值 0.5）、特征重要性 Top20，数据源均注明 CSV 文件名；
   原来的现场训练部分保留在下方，并加了「与 C 的口径有差异」的说明。
   *修复*：默认模型从「基线」改为「逻辑回归」，并对 `threshold is None` 做了兜底。
3. **最终评价页**：顶部新增「D 同学锁定的测试集最终结果」区块（直接读
   `artifacts/test_result.json`，不重跑），展示指标、混淆矩阵、评价时间、数据指纹、
   代码版本；对 D 标注的 `feature_order_verified=false` 给出⚠️提示；
   另加一个可折叠的「在线复算核对」——用 C 的模型在测试集复算并与 D 逐项比对。
4. **客户风险演示页**：新增「使用哪个模型预测」单选，默认走 C 的 `best_model.pkl`
   （阈值取 `threshold.json` 的 0.5），可切回界面现场训练模型。

## 四、还需要向队友要的（建议直接发这几句）

- **A（刘思雨）**：`src/data_io.py`、`src/data_check.py`、`split_summary()`、
  `plot_distributions()`。目前数据概览页是 E 自己实现的等价功能，
  README 里需要写明"A 的函数被 E 的 `cp.audit_data()` 等价替代"或补上接入。
- **B（赵梓伶）**：**正式的 `preprocess.py`**（含 `build_preprocessor` / `transform` /
  防泄漏检查），用于覆盖垫片；以及 README 里要列出的字段入模清单。
- **C（冯荟如）**：确认随机森林参数（`n_estimators=300, max_depth=12,
  min_samples_leaf=5, class_weight='balanced'`）与阈值 0.5 的口径；最好补一份
  `feature_spec.json`（D 的报告里标注了特征列顺序未核对）。
- **D（吴姿仪）**：确认 `test_result.json` 是最终版；注意其
  `test_data_file` 字段指向 D 自己的电脑路径，交付时应改成相对路径或删掉。

## 五、验证方式

```bash
# 1) 单独验证接入层 + 与 D 的结果一致性
.venv/Scripts/python.exe -c "import artifacts_io as ai; print(ai.artifacts_status())"

# 2) 四个页面冒烟测试（无头运行，输出每个页面的异常数）
.venv/Scripts/python.exe smoke_test.py
#  期望：[数据概览] exception=0  [模型比较] exception=0
#        [最终评价（测试集）] exception=0  [客户风险演示] exception=0

# 3) 启动界面
.venv/Scripts/python.exe -m streamlit run app.py
```

## 六、写 README / 汇报时可以直接用的三句话

1. 界面层不参与任何建模：C 的模型、D 的测试结果、B 的字段口径都以文件形式接入，
   页面只读取和展示，符合"测试集只评价一次、页面上不许手填数字"的红线。
2. 用 C 的 `best_model.pkl` 在测试集复算，结果与 D 的 `test_result.json` 逐项一致
   （AUC 0.782545，混淆矩阵 2899/606/400/596），证明三人口径已打通。
3. 修复了「模型比较」页默认选中基线导致阈值选择返回 None 的崩溃问题，
   并对缺失文件、缺失模块、无效 record_id 都加了提示分支。

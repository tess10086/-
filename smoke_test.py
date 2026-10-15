import pandas as pd
from streamlit.testing.v1 import AppTest

df = pd.read_csv("01_credit_default.csv", encoding="utf-8-sig")
te = df[df["split"] == "test"]
rid = int(te.iloc[0]["record_id"])
print("test record_id =", rid)

for p in ["数据概览", "模型比较", "最终评价（测试集）", "客户风险演示"]:
    at = AppTest.from_file("app.py", default_timeout=600).run()
    at.sidebar.radio[0].set_value(p)
    at.run()
    errs = [e.value for e in at.exception]
    print(f"[{p}] exception={len(errs)}", errs[:1])

# 客户风险演示：用 C 同学模型预测一条测试集记录
at = AppTest.from_file("app.py", default_timeout=600).run()
at.sidebar.radio[0].set_value("客户风险演示")
at.run()
src_radio = [r for r in at.radio if r.label == "使用哪个模型预测"]
print("模型来源选项 =", src_radio[0].options if src_radio else None)
src_radio[0].set_value(src_radio[0].options[0])
at.run()
at.number_input[0].set_value(rid)
btn = [b for b in at.button if "查询" in b.label][0]
btn.click()
at.run()
errs = [e.value for e in at.exception]
print("[演示- C模型] exception=", len(errs), errs[:1])
print("markdown 含风险概率 =", any("风险概率" in md.value for md in at.markdown))

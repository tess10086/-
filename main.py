
import streamlit as st

st.set_page_config(page_title="我的第一个 Streamlit 应用", page_icon="📊")

st.title("Hello Streamlit")
st.write("这是一个最简单的 Streamlit 程序。")

name = st.text_input("请输入你的名字")

if name:
    st.success(f"你好，{name}！")

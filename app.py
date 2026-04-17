import streamlit as st

st.title("ECD Boltzmann Averaging App")
st.write("Gaussian の log/out ファイルを読み込む練習")

uploaded_files = st.file_uploader(
    "Gaussian の .log または .out ファイルを選択してください",
    type=["log", "out"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"アップロードされたファイル数: {len(uploaded_files)}")

    for f in uploaded_files:
        st.subheader(f.name)

        text = f.read().decode("utf-8", errors="ignore")

        st.write("先頭500文字")
        st.text(text[:500])
else:
    st.info("まだファイルが選択されていません。")

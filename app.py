import re
import streamlit as st

st.title("ECD Boltzmann Averaging App")
st.write("Gaussian の log/out ファイルからエネルギーを抜き出す練習")

def extract_energies(text):
    result = {
        "scf_energy": None,
        "zpe_energy": None,
        "free_energy": None,
    }

    # SCF Done の最後の値を使う
    scf_matches = re.findall(
        r"SCF Done:\s+E\([^)]+\)\s+=\s+(-?\d+\.\d+)",
        text
    )
    if scf_matches:
        result["scf_energy"] = float(scf_matches[-1])

    # Zero-point energy
    zpe_matches = re.findall(
        r"Sum of electronic and zero-point Energies=\s+(-?\d+\.\d+)",
        text
    )
    if zpe_matches:
        result["zpe_energy"] = float(zpe_matches[-1])

    # Free energy
    free_matches = re.findall(
        r"Sum of electronic and thermal Free Energies=\s+(-?\d+\.\d+)",
        text
    )
    if free_matches:
        result["free_energy"] = float(free_matches[-1])

    return result


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
        energies = extract_energies(text)

        st.write("**抽出結果**")
        st.write(f"SCF energy: {energies['scf_energy']}")
        st.write(f"Zero-point energy: {energies['zpe_energy']}")
        st.write(f"Free energy: {energies['free_energy']}")

        with st.expander("ログ先頭1000文字を表示"):
            st.text(text[:1000])

else:
    st.info("まだファイルが選択されていません。")

import re
import math
import pandas as pd
import streamlit as st

st.title("ECD Boltzmann Averaging App")
st.write("Gaussian の log/out ファイルからエネルギーを抽出し、ボルツマン重みを計算します。")

HARTREE_TO_KCAL = 627.509474
R_KCAL = 0.0019872041  # kcal mol^-1 K^-1


def extract_energies(text):
    result = {
        "scf_energy": None,
        "zpe_energy": None,
        "free_energy": None,
    }

    scf_matches = re.findall(
        r"SCF Done:\s+E\([^)]+\)\s+=\s+(-?\d+\.\d+)",
        text
    )
    if scf_matches:
        result["scf_energy"] = float(scf_matches[-1])

    zpe_matches = re.findall(
        r"Sum of electronic and zero-point Energies=\s+(-?\d+\.\d+)",
        text
    )
    if zpe_matches:
        result["zpe_energy"] = float(zpe_matches[-1])

    free_matches = re.findall(
        r"Sum of electronic and thermal Free Energies=\s+(-?\d+\.\d+)",
        text
    )
    if free_matches:
        result["free_energy"] = float(free_matches[-1])

    return result


def safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return 0.0


uploaded_files = st.file_uploader(
    "Gaussian の .log または .out ファイルを選択してください",
    type=["log", "out"],
    accept_multiple_files=True
)

energy_choice = st.selectbox(
    "ボルツマン平均に使うエネルギーを選んでください",
    options=[
        "free_energy",
        "zpe_energy",
        "scf_energy",
    ],
    format_func=lambda x: {
        "free_energy": "Free energy",
        "zpe_energy": "Zero-point energy",
        "scf_energy": "SCF energy",
    }[x]
)

temperature = st.number_input(
    "温度 (K)",
    min_value=1.0,
    value=298.15,
    step=1.0
)

if uploaded_files:
    records = []

    for f in uploaded_files:
        text = f.read().decode("utf-8", errors="ignore")
        energies = extract_energies(text)

        records.append({
            "file_name": f.name,
            "scf_energy": energies["scf_energy"],
            "zpe_energy": energies["zpe_energy"],
            "free_energy": energies["free_energy"],
        })

    df = pd.DataFrame(records)

    st.subheader("抽出されたエネルギー")
    st.dataframe(df)

    # 選択したエネルギーがある行だけ残す
    valid_df = df[df[energy_choice].notna()].copy()

    if valid_df.empty:
        st.error("選択したエネルギーがどのファイルからも取得できませんでした。")
    else:
        # 最低エネルギーを基準に相対エネルギーを計算
        e_min = valid_df[energy_choice].min()
        valid_df["delta_E_hartree"] = valid_df[energy_choice] - e_min
        valid_df["delta_E_kcal_mol"] = valid_df["delta_E_hartree"] * HARTREE_TO_KCAL

        # ボルツマン因子
        valid_df["boltz_factor"] = valid_df["delta_E_kcal_mol"].apply(
            lambda x: safe_exp(-x / (R_KCAL * temperature))
        )

        factor_sum = valid_df["boltz_factor"].sum()

        if factor_sum == 0:
            st.error("ボルツマン因子の合計が 0 になりました。エネルギー差や温度を確認してください。")
        else:
            valid_df["boltz_weight"] = valid_df["boltz_factor"] / factor_sum

            # 表示用に並べ替え
            valid_df = valid_df.sort_values(by=energy_choice).reset_index(drop=True)

            st.subheader("相対エネルギーとボルツマン重み")
            st.dataframe(valid_df)

            st.write("### 確認")
            st.write(f"採用エネルギー: {energy_choice}")
            st.write(f"基準エネルギー: {e_min:.8f} Hartree")
            st.write(f"重みの合計: {valid_df['boltz_weight'].sum():.6f}")

else:
    st.info("まだファイルが選択されていません。")

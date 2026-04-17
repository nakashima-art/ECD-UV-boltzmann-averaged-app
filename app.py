import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("ECD Boltzmann Averaging App")
st.write("Gaussian TD-DFT ログから ECD スペクトルを作成し、ボルツマン平均します。")

HARTREE_TO_KCAL = 627.509474
R_KCAL = 0.0019872041  # kcal mol^-1 K^-1


# -----------------------------
# エネルギー抽出
# -----------------------------
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


# -----------------------------
# Rotatory strength 抽出
# Gaussian の "R(length)" 行を想定
# -----------------------------
def extract_rotatory_strengths(text):
    rot_strengths = []

    # 例:
    # 1/2=  -0.1234  0.5678 ...
    # ではなく、Gaussian の
    # R(length) などを含む部分は出力形式がやや揺れるため、
    # まず "Rotatory Strengths" セクションを拾わずに、
    # よく出るパターンを直接探す
    #
    # 代表例:
    # 1  -0.1234
    # 2   0.5678
    #
    # 今回は "state番号 + 数値" を拾う簡易版ではなく、
    # "R(length)" 行からすべての数値を取る方式にします。

    # Gaussian 出力によっては "Rotatory Strengths (R)" の後に
    # 数表が続く場合があるので、そのブロックをざっくり抽出
    block_match = re.search(
        r"Rotatory Strengths.*?\n(.*?)(?:\n\s*\n|\n\s*Total|\Z)",
        text,
        re.DOTALL
    )

    if not block_match:
        return rot_strengths

    block = block_match.group(1)

    # 行ごとに処理
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue

        # 例:
        # 1   -0.12345
        # 2    0.23456
        # または複数列ある場合もあるので、先頭が整数の行だけ見る
        if re.match(r"^\d+", line):
            nums = re.findall(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?", line)
            if nums:
                # 一番最後の数値を rotatory strength とみなす簡易版
                rot_strengths.append(float(nums[-1]))

    return rot_strengths


# -----------------------------
# Excited State 抽出
# 例:
# Excited State   1:      Singlet-A      4.1234 eV  300.73 nm  f=0.1234
# -----------------------------
def extract_excited_states(text):
    states = []

    pattern = re.compile(
        r"Excited State\s+(\d+):.*?(\d+\.\d+)\s+eV\s+(\d+\.\d+)\s+nm\s+f=([-\d\.]+)",
        re.MULTILINE
    )

    for m in pattern.finditer(text):
        state_num = int(m.group(1))
        excitation_ev = float(m.group(2))
        wavelength_nm = float(m.group(3))
        oscillator_strength = float(m.group(4))

        states.append({
            "state": state_num,
            "excitation_ev": excitation_ev,
            "wavelength_nm": wavelength_nm,
            "osc_strength": oscillator_strength,
        })

    return states


# -----------------------------
# excited states と rotatory strengths を結合
# -----------------------------
def extract_transitions(text):
    states = extract_excited_states(text)
    rot_strengths = extract_rotatory_strengths(text)

    transitions = []

    n = min(len(states), len(rot_strengths))
    for i in range(n):
        row = states[i].copy()
        row["rot_strength"] = rot_strengths[i]
        transitions.append(row)

    return transitions


# -----------------------------
# ボルツマン計算
# -----------------------------
def safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return 0.0


def build_energy_table(file_data_list, energy_choice, temperature):
    df = pd.DataFrame(file_data_list)

    valid_df = df[df[energy_choice].notna()].copy()

    if valid_df.empty:
        return df, None

    e_min = valid_df[energy_choice].min()
    valid_df["delta_E_hartree"] = valid_df[energy_choice] - e_min
    valid_df["delta_E_kcal_mol"] = valid_df["delta_E_hartree"] * HARTREE_TO_KCAL

    valid_df["boltz_factor"] = valid_df["delta_E_kcal_mol"].apply(
        lambda x: safe_exp(-x / (R_KCAL * temperature))
    )

    factor_sum = valid_df["boltz_factor"].sum()
    if factor_sum == 0:
        return df, None

    valid_df["boltz_weight"] = valid_df["boltz_factor"] / factor_sum
    valid_df = valid_df.sort_values(by=energy_choice).reset_index(drop=True)

    return df, valid_df


# -----------------------------
# Gaussian broadening
# -----------------------------
def gaussian_broadening(x, center, height, sigma):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def build_ecd_spectrum(transitions, wavelength_grid, sigma_nm):
    y = np.zeros_like(wavelength_grid)

    for tr in transitions:
        wl = tr["wavelength_nm"]
        rot = tr["rot_strength"]
        y += gaussian_broadening(wavelength_grid, wl, rot, sigma_nm)

    return y


# -----------------------------
# UI
# -----------------------------
uploaded_files = st.file_uploader(
    "Gaussian の .log または .out ファイルを選択してください",
    type=["log", "out"],
    accept_multiple_files=True
)

energy_choice = st.selectbox(
    "ボルツマン平均に使うエネルギーを選んでください",
    options=["free_energy", "zpe_energy", "scf_energy"],
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

st.subheader("スペクトル設定")
wl_min = st.number_input("最小波長 (nm)", value=180.0)
wl_max = st.number_input("最大波長 (nm)", value=450.0)
n_points = st.number_input("プロット点数", min_value=200, value=2000, step=100)
sigma_nm = st.number_input("Gaussian broadening 幅 sigma (nm)", min_value=0.1, value=10.0, step=0.5)

if uploaded_files:
    file_data_list = []
    transition_tables = {}

    for f in uploaded_files:
        text = f.read().decode("utf-8", errors="ignore")

        energies = extract_energies(text)
        transitions = extract_transitions(text)

        file_data_list.append({
            "file_name": f.name,
            "scf_energy": energies["scf_energy"],
            "zpe_energy": energies["zpe_energy"],
            "free_energy": energies["free_energy"],
            "n_transitions": len(transitions),
        })

        transition_tables[f.name] = transitions

    raw_df, valid_df = build_energy_table(file_data_list, energy_choice, temperature)

    st.subheader("抽出されたエネルギー")
    st.dataframe(raw_df)

    if valid_df is None:
        st.error("選択したエネルギーでボルツマン重みを計算できませんでした。")
    else:
        st.subheader("相対エネルギーとボルツマン重み")
        st.dataframe(valid_df)

        # 波長軸
        wavelength_grid = np.linspace(wl_min, wl_max, int(n_points))

        # 各配座スペクトル
        individual_spectra = {}
        averaged_spectrum = np.zeros_like(wavelength_grid)

        for _, row in valid_df.iterrows():
            file_name = row["file_name"]
            weight = row["boltz_weight"]
            transitions = transition_tables.get(file_name, [])

            if len(transitions) == 0:
                continue

            y = build_ecd_spectrum(transitions, wavelength_grid, sigma_nm)
            individual_spectra[file_name] = y
            averaged_spectrum += weight * y

        # 遷移情報表示
        st.subheader("抽出された TD-DFT 遷移")
        for file_name, transitions in transition_tables.items():
            with st.expander(f"{file_name} の遷移を表示"):
                if transitions:
                    st.dataframe(pd.DataFrame(transitions))
                else:
                    st.warning("遷移情報または rotatory strength を抽出できませんでした。")

        # グラフ
        st.subheader("ECD スペクトル")

        show_individual = st.checkbox("各配座スペクトルも表示する", value=True)

        fig, ax = plt.subplots(figsize=(8, 5))

        if show_individual:
            for file_name, y in individual_spectra.items():
                ax.plot(wavelength_grid, y, label=file_name)

        ax.plot(wavelength_grid, averaged_spectrum, linewidth=2.5, label="Boltzmann-averaged ECD")
        ax.axhline(0, linewidth=1)

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("ECD intensity (arb. units)")
        ax.set_title("ECD Spectrum")
        ax.legend()

        st.pyplot(fig)

        # CSV 出力
        export_df = pd.DataFrame({
            "wavelength_nm": wavelength_grid,
            "ecd_avg": averaged_spectrum,
        })

        for file_name, y in individual_spectra.items():
            export_df[f"ecd_{file_name}"] = y

        csv_data = export_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "平均スペクトルCSVをダウンロード",
            data=csv_data,
            file_name="ecd_boltzmann_averaged.csv",
            mime="text/csv"
        )

else:
    st.info("まだファイルが選択されていません。")

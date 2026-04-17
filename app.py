import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("ECD Boltzmann Averaging App")
st.write("opt/optfreq ログと TD-DFT ログを別々に読み込み、配座ごとに対応付けて ECD をボルツマン平均します。")

HARTREE_TO_KCAL = 627.509474
R_KCAL = 0.0019872041  # kcal mol^-1 K^-1


# --------------------------------------------------
# 補助：ファイル名から配座キーを作る
# 例:
# conf01_optfreq.log -> conf01
# conf01_tddft.log   -> conf01
# --------------------------------------------------
def make_base_key(filename):
    name = filename.rsplit(".", 1)[0]  # 拡張子除去
    name = re.sub(r"(_optfreq|_opt|_freq|_tddft|_td)$", "", name, flags=re.IGNORECASE)
    return name


# --------------------------------------------------
# エネルギー抽出（opt/optfreq側）
# --------------------------------------------------
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


# --------------------------------------------------
# TD-DFT 遷移抽出
# 例:
# Excited State   1:      Singlet-A      4.1234 eV  300.73 nm  f=0.1234
# --------------------------------------------------
def extract_excited_states(text):
    states = []

    pattern = re.compile(
        r"Excited State\s+(\d+):.*?(\d+\.\d+)\s+eV\s+(\d+\.\d+)\s+nm\s+f=([-\d\.]+)",
        re.MULTILINE
    )

    for m in pattern.finditer(text):
        states.append({
            "state": int(m.group(1)),
            "excitation_ev": float(m.group(2)),
            "wavelength_nm": float(m.group(3)),
            "osc_strength": float(m.group(4)),
        })

    return states


# --------------------------------------------------
# Rotatory strength 抽出
# 注意:
# Gaussian の出力形式は揺れるので、まずは簡易版
# "Rotatory Strengths" ブロックの各行末の数値を採用
# --------------------------------------------------
def extract_rotatory_strengths(text):
    rot_strengths = []

    block_match = re.search(
        r"Rotatory Strengths.*?\n(.*?)(?:\n\s*\n|\n\s*Total|\Z)",
        text,
        re.DOTALL
    )

    if not block_match:
        return rot_strengths

    block = block_match.group(1)

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue

        if re.match(r"^\d+", line):
            nums = re.findall(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?", line)
            if nums:
                rot_strengths.append(float(nums[-1]))

    return rot_strengths


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


# --------------------------------------------------
# ボルツマン計算
# --------------------------------------------------
def safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return 0.0


def gaussian_broadening(x, center, height, sigma):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def build_ecd_spectrum(transitions, wavelength_grid, sigma_nm):
    y = np.zeros_like(wavelength_grid)

    for tr in transitions:
        wl = tr["wavelength_nm"]
        rot = tr["rot_strength"]
        y += gaussian_broadening(wavelength_grid, wl, rot, sigma_nm)

    return y


# --------------------------------------------------
# UI
# --------------------------------------------------
st.subheader("1. opt / optfreq ファイル")
optfreq_files = st.file_uploader(
    "opt または optfreq の .log / .out を選択してください",
    type=["log", "out"],
    accept_multiple_files=True,
    key="optfreq_files"
)

st.subheader("2. TD-DFT ファイル")
tddft_files = st.file_uploader(
    "TD-DFT の .log / .out を選択してください",
    type=["log", "out"],
    accept_multiple_files=True,
    key="tddft_files"
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

st.subheader("3. スペクトル設定")
wl_min = st.number_input("最小波長 (nm)", value=180.0)
wl_max = st.number_input("最大波長 (nm)", value=450.0)
n_points = st.number_input("プロット点数", min_value=200, value=2000, step=100)
sigma_nm = st.number_input("Gaussian broadening 幅 sigma (nm)", min_value=0.1, value=10.0, step=0.5)

if optfreq_files and tddft_files:
    # ------------------------
    # optfreq 側を辞書化
    # ------------------------
    energy_dict = {}
    for f in optfreq_files:
        text = f.read().decode("utf-8", errors="ignore")
        key = make_base_key(f.name)
        energies = extract_energies(text)

        energy_dict[key] = {
            "optfreq_file": f.name,
            "scf_energy": energies["scf_energy"],
            "zpe_energy": energies["zpe_energy"],
            "free_energy": energies["free_energy"],
        }

    # ------------------------
    # tddft 側を辞書化
    # ------------------------
    transition_dict = {}
    for f in tddft_files:
        text = f.read().decode("utf-8", errors="ignore")
        key = make_base_key(f.name)
        transitions = extract_transitions(text)

        transition_dict[key] = {
            "tddft_file": f.name,
            "transitions": transitions,
            "n_transitions": len(transitions),
        }

    # ------------------------
    # 共通キーでマージ
    # ------------------------
    common_keys = sorted(set(energy_dict.keys()) & set(transition_dict.keys()))
    only_energy_keys = sorted(set(energy_dict.keys()) - set(transition_dict.keys()))
    only_tddft_keys = sorted(set(transition_dict.keys()) - set(energy_dict.keys()))

    st.subheader("4. ペアリング結果")
    st.write(f"対応付けできた配座数: {len(common_keys)}")

    if only_energy_keys:
        st.warning("opt/optfreq 側だけにあるキー: " + ", ".join(only_energy_keys))

    if only_tddft_keys:
        st.warning("TD-DFT 側だけにあるキー: " + ", ".join(only_tddft_keys))

    if len(common_keys) == 0:
        st.error("共通の配座キーが見つかりませんでした。ファイル名規則を確認してください。")
    else:
        records = []
        transition_tables = {}

        for key in common_keys:
            e = energy_dict[key]
            t = transition_dict[key]

            records.append({
                "conf_key": key,
                "optfreq_file": e["optfreq_file"],
                "tddft_file": t["tddft_file"],
                "scf_energy": e["scf_energy"],
                "zpe_energy": e["zpe_energy"],
                "free_energy": e["free_energy"],
                "n_transitions": t["n_transitions"],
            })

            transition_tables[key] = t["transitions"]

        df = pd.DataFrame(records)

        st.subheader("5. 対応付け後の一覧")
        st.dataframe(df)

        valid_df = df[df[energy_choice].notna()].copy()

        if valid_df.empty:
            st.error("選択したエネルギーが取得できた配座がありません。")
        else:
            e_min = valid_df[energy_choice].min()
            valid_df["delta_E_hartree"] = valid_df[energy_choice] - e_min
            valid_df["delta_E_kcal_mol"] = valid_df["delta_E_hartree"] * HARTREE_TO_KCAL
            valid_df["boltz_factor"] = valid_df["delta_E_kcal_mol"].apply(
                lambda x: safe_exp(-x / (R_KCAL * temperature))
            )

            factor_sum = valid_df["boltz_factor"].sum()

            if factor_sum == 0:
                st.error("ボルツマン因子の合計が 0 になりました。")
            else:
                valid_df["boltz_weight"] = valid_df["boltz_factor"] / factor_sum
                valid_df = valid_df.sort_values(by=energy_choice).reset_index(drop=True)

                st.subheader("6. 相対エネルギーとボルツマン重み")
                st.dataframe(valid_df)

                wavelength_grid = np.linspace(wl_min, wl_max, int(n_points))
                individual_spectra = {}
                averaged_spectrum = np.zeros_like(wavelength_grid)

                for _, row in valid_df.iterrows():
                    key = row["conf_key"]
                    weight = row["boltz_weight"]
                    transitions = transition_tables.get(key, [])

                    if len(transitions) == 0:
                        continue

                    y = build_ecd_spectrum(transitions, wavelength_grid, sigma_nm)
                    individual_spectra[key] = y
                    averaged_spectrum += weight * y

                st.subheader("7. 抽出された TD-DFT 遷移")
                for key in common_keys:
                    with st.expander(f"{key} の遷移を表示"):
                        transitions = transition_tables.get(key, [])
                        if transitions:
                            st.dataframe(pd.DataFrame(transitions))
                        else:
                            st.warning("遷移情報または rotatory strength を抽出できませんでした。")

                st.subheader("8. ECD スペクトル")
                show_individual = st.checkbox("各配座スペクトルも表示する", value=True)

                fig, ax = plt.subplots(figsize=(8, 5))

                if show_individual:
                    for key, y in individual_spectra.items():
                        ax.plot(wavelength_grid, y, label=key)

                ax.plot(wavelength_grid, averaged_spectrum, linewidth=2.5, label="Boltzmann-averaged ECD")
                ax.axhline(0, linewidth=1)

                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("ECD intensity (arb. units)")
                ax.set_title("ECD Spectrum")
                ax.legend()

                st.pyplot(fig)

                export_df = pd.DataFrame({
                    "wavelength_nm": wavelength_grid,
                    "ecd_avg": averaged_spectrum,
                })

                for key, y in individual_spectra.items():
                    export_df[f"ecd_{key}"] = y

                csv_data = export_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "平均スペクトルCSVをダウンロード",
                    data=csv_data,
                    file_name="ecd_boltzmann_averaged.csv",
                    mime="text/csv"
                )

else:
    st.info("opt/optfreq ファイル群と TD-DFT ファイル群の両方をアップロードしてください。")

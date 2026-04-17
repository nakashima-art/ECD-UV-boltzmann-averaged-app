import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("ECD / UV Boltzmann Averaging App")
st.write("opt/optfreq ログと TD-DFT ログを別々に読み込み、配座ごとに対応付けて UV / ECD をボルツマン平均します。")

HARTREE_TO_KCAL = 627.509474
R_KCAL = 0.0019872041  # kcal mol^-1 K^-1
NM_PER_EV = 1239.841984  # lambda(nm) = 1239.841984 / E(eV)


# --------------------------------------------------
# 補助：ファイル名正規化
# --------------------------------------------------
def normalize_filename_for_pairing(filename):
    name = filename.rsplit(".", 1)[0]
    name = name.lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"(?:_)?(?:optfreq|opt|freq|tddft|td)(?:_)?", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def split_tokens_for_suffix_matching(filename):
    normalized = normalize_filename_for_pairing(filename)
    if not normalized:
        return []
    return normalized.split("_")


def common_suffix_token_count(tokens1, tokens2):
    n = 0
    i = 1
    while i <= min(len(tokens1), len(tokens2)):
        if tokens1[-i] == tokens2[-i]:
            n += 1
            i += 1
        else:
            break
    return n


def common_suffix_char_length(s1, s2):
    n = 0
    i = 1
    while i <= min(len(s1), len(s2)):
        if s1[-i] == s2[-i]:
            n += 1
            i += 1
        else:
            break
    return n


def pair_files_by_common_suffix(optfreq_files, tddft_files):
    opt_infos = []
    for f in optfreq_files:
        opt_infos.append({
            "file": f,
            "name": f.name,
            "norm": normalize_filename_for_pairing(f.name),
            "tokens": split_tokens_for_suffix_matching(f.name),
        })

    td_infos = []
    for f in tddft_files:
        td_infos.append({
            "file": f,
            "name": f.name,
            "norm": normalize_filename_for_pairing(f.name),
            "tokens": split_tokens_for_suffix_matching(f.name),
        })

    candidate_pairs = []
    for oi in opt_infos:
        for ti in td_infos:
            token_score = common_suffix_token_count(oi["tokens"], ti["tokens"])
            char_score = common_suffix_char_length(oi["norm"], ti["norm"])

            if token_score == 0 and char_score < 3:
                continue

            candidate_pairs.append({
                "opt_name": oi["name"],
                "td_name": ti["name"],
                "token_score": token_score,
                "char_score": char_score,
            })

    candidate_pairs.sort(
        key=lambda x: (x["token_score"], x["char_score"]),
        reverse=True
    )

    used_opt = set()
    used_td = set()
    final_pairs = []

    for c in candidate_pairs:
        if c["opt_name"] in used_opt:
            continue
        if c["td_name"] in used_td:
            continue

        used_opt.add(c["opt_name"])
        used_td.add(c["td_name"])
        final_pairs.append(c)

    return final_pairs


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
# --------------------------------------------------
def extract_rotatory_strengths(text, mode="length"):
    """
    mode:
        "length"   -> R(length) を抽出
        "velocity" -> R(velocity) を抽出
    """
    rot_strengths = []

    if mode == "length":
        header_pattern = (
            r"Rotatory Strengths \(R\) in cgs .*?\n"
            r"\s*state\s+XX\s+YY\s+ZZ\s+R\(length\)\s*\n"
        )
    elif mode == "velocity":
        header_pattern = (
            r"Rotatory Strengths \(R\) in cgs .*?\n"
            r"\s*state\s+XX\s+YY\s+ZZ\s+R\(velocity\)\s+E-M Angle\s*\n"
        )
    else:
        raise ValueError("mode must be 'length' or 'velocity'")

    m = re.search(header_pattern, text, re.IGNORECASE)
    if not m:
        return rot_strengths

    start = m.end()
    lines = text[start:].splitlines()

    for line in lines:
        stripped = line.strip()

        if not stripped:
            break

        if not re.match(r"^\d+", stripped):
            break

        parts = stripped.split()

        try:
            if mode == "length" and len(parts) >= 5:
                rot_strengths.append(float(parts[4]))
            elif mode == "velocity" and len(parts) >= 5:
                rot_strengths.append(float(parts[4]))
        except ValueError:
            continue

    return rot_strengths


def extract_transitions(text):
    states = extract_excited_states(text)
    rot_strengths_len = extract_rotatory_strengths(text, mode="length")
    rot_strengths_vel = extract_rotatory_strengths(text, mode="velocity")

    transitions = []
    n = min(len(states), len(rot_strengths_len))

    for i in range(n):
        row = states[i].copy()
        row["rot_strength"] = rot_strengths_len[i]  # ECD 描画に使用
        row["rot_strength_length"] = rot_strengths_len[i]
        row["rot_strength_velocity"] = (
            rot_strengths_vel[i] if i < len(rot_strengths_vel) else None
        )
        transitions.append(row)

    return transitions


# --------------------------------------------------
# 数学補助
# --------------------------------------------------
def safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return 0.0


def gaussian_broadening(x, center, height, sigma):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def halfwidth_to_sigma(value):
    # Gaussian の半値半幅 HWHM -> sigma
    # HWHM = sigma * sqrt(2 ln 2)
    return value / math.sqrt(2.0 * math.log(2.0))


def make_wavelength_grid(wl_min, wl_max, axis_mode, point_spacing_nm=None, n_points=None):
    if axis_mode == "point_spacing":
        if point_spacing_nm is None or point_spacing_nm <= 0:
            raise ValueError("point_spacing_nm must be positive")
        return np.arange(wl_min, wl_max + point_spacing_nm * 0.5, point_spacing_nm)
    elif axis_mode == "n_points":
        if n_points is None or n_points < 2:
            raise ValueError("n_points must be >= 2")
        return np.linspace(wl_min, wl_max, int(n_points))
    else:
        raise ValueError("axis_mode must be 'point_spacing' or 'n_points'")


# --------------------------------------------------
# スペクトル構築
# --------------------------------------------------
def build_spectrum_nm(transitions, wavelength_grid, intensity_key, sigma_nm):
    y = np.zeros_like(wavelength_grid)

    for tr in transitions:
        wl = tr["wavelength_nm"]
        height = tr.get(intensity_key, None)
        if height is None:
            continue
        y += gaussian_broadening(wavelength_grid, wl, height, sigma_nm)

    return y


def build_spectrum_ev(transitions, wavelength_grid_nm, intensity_key, sigma_ev):
    # エネルギー空間で broadening してから、波長軸に補間
    e_min = NM_PER_EV / np.max(wavelength_grid_nm)
    e_max = NM_PER_EV / np.min(wavelength_grid_nm)

    # 十分細かい内部グリッド
    e_grid = np.linspace(e_min, e_max, 4000)
    y_e = np.zeros_like(e_grid)

    for tr in transitions:
        e0 = tr["excitation_ev"]
        height = tr.get(intensity_key, None)
        if height is None:
            continue
        y_e += gaussian_broadening(e_grid, e0, height, sigma_ev)

    wl_from_e = NM_PER_EV / e_grid

    order = np.argsort(wl_from_e)
    wl_sorted = wl_from_e[order]
    y_sorted = y_e[order]

    y_nm = np.interp(wavelength_grid_nm, wl_sorted, y_sorted, left=0.0, right=0.0)
    return y_nm


def build_uv_spectrum(transitions, wavelength_grid, broadening_mode, sigma_nm=None, halfwidth_ev=None):
    if broadening_mode == "sigma_nm":
        return build_spectrum_nm(transitions, wavelength_grid, "osc_strength", sigma_nm)
    elif broadening_mode == "halfwidth_ev":
        sigma_ev = halfwidth_to_sigma(halfwidth_ev)
        return build_spectrum_ev(transitions, wavelength_grid, "osc_strength", sigma_ev)
    else:
        raise ValueError("Unknown broadening_mode")


def build_ecd_spectrum(transitions, wavelength_grid, broadening_mode, sigma_nm=None, halfwidth_ev=None):
    if broadening_mode == "sigma_nm":
        return build_spectrum_nm(transitions, wavelength_grid, "rot_strength", sigma_nm)
    elif broadening_mode == "halfwidth_ev":
        sigma_ev = halfwidth_to_sigma(halfwidth_ev)
        return build_spectrum_ev(transitions, wavelength_grid, "rot_strength", sigma_ev)
    else:
        raise ValueError("Unknown broadening_mode")


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

axis_mode = st.radio(
    "横軸の指定方法",
    options=["point_spacing", "n_points"],
    format_func=lambda x: {
        "point_spacing": "ポイント幅で指定",
        "n_points": "ポイント点数で指定",
    }[x]
)

if axis_mode == "point_spacing":
    point_spacing_nm = st.number_input(
        "ポイント幅 (nm)",
        min_value=0.001,
        value=0.2,
        step=0.1,
        format="%.4f"
    )
    n_points = None
else:
    n_points = st.number_input(
        "プロット点数",
        min_value=2,
        value=2000,
        step=100
    )
    point_spacing_nm = None

broadening_mode = st.radio(
    "Broadening の指定方法",
    options=["sigma_nm", "halfwidth_ev"],
    format_func=lambda x: {
        "sigma_nm": "Gaussian broadening 幅 sigma (nm)",
        "halfwidth_ev": "Half-Width (eV)",
    }[x]
)

if broadening_mode == "sigma_nm":
    sigma_nm = st.number_input(
        "Gaussian broadening 幅 sigma (nm)",
        min_value=0.001,
        value=10.0,
        step=0.5
    )
    halfwidth_ev = None
else:
    halfwidth_ev = st.number_input(
        "Half-Width (eV)",
        min_value=0.0001,
        value=0.10,
        step=0.01,
        format="%.4f"
    )
    sigma_nm = None

if optfreq_files and tddft_files:
    optfreq_data = {}
    for f in optfreq_files:
        text = f.read().decode("utf-8", errors="ignore")
        energies = extract_energies(text)
        optfreq_data[f.name] = {
            "optfreq_file": f.name,
            "scf_energy": energies["scf_energy"],
            "zpe_energy": energies["zpe_energy"],
            "free_energy": energies["free_energy"],
        }

    tddft_data = {}
    for f in tddft_files:
        text = f.read().decode("utf-8", errors="ignore")
        transitions = extract_transitions(text)
        tddft_data[f.name] = {
            "tddft_file": f.name,
            "transitions": transitions,
            "n_transitions": len(transitions),
        }

    pairs = pair_files_by_common_suffix(optfreq_files, tddft_files)

    paired_opt_names = {p["opt_name"] for p in pairs}
    paired_td_names = {p["td_name"] for p in pairs}

    only_energy_keys = sorted(set(optfreq_data.keys()) - paired_opt_names)
    only_tddft_keys = sorted(set(tddft_data.keys()) - paired_td_names)

    st.subheader("4. ペアリング結果")
    st.write(f"対応付けできた配座数: {len(pairs)}")

    if only_energy_keys:
        st.warning("opt/optfreq 側だけにあるファイル: " + ", ".join(only_energy_keys))

    if only_tddft_keys:
        st.warning("TD-DFT 側だけにあるファイル: " + ", ".join(only_tddft_keys))

    if len(pairs) == 0:
        st.error("対応付け可能なファイルペアが見つかりませんでした。ファイル名規則を確認してください。")
    else:
        records = []
        transition_tables = {}

        for i, pair in enumerate(pairs, start=1):
            opt_name = pair["opt_name"]
            td_name = pair["td_name"]

            e = optfreq_data[opt_name]
            t = tddft_data[td_name]

            conf_label = f"pair_{i:02d}"

            records.append({
                "conf_key": conf_label,
                "optfreq_file": e["optfreq_file"],
                "tddft_file": t["tddft_file"],
                "suffix_token_score": pair["token_score"],
                "suffix_char_score": pair["char_score"],
                "scf_energy": e["scf_energy"],
                "zpe_energy": e["zpe_energy"],
                "free_energy": e["free_energy"],
                "n_transitions": t["n_transitions"],
            })

            transition_tables[conf_label] = t["transitions"]

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

                wavelength_grid = make_wavelength_grid(
                    wl_min=wl_min,
                    wl_max=wl_max,
                    axis_mode=axis_mode,
                    point_spacing_nm=point_spacing_nm,
                    n_points=n_points
                )

                st.write(f"実際の横軸点数: {len(wavelength_grid)}")

                individual_uv_spectra = {}
                individual_ecd_spectra = {}
                averaged_uv_spectrum = np.zeros_like(wavelength_grid)
                averaged_ecd_spectrum = np.zeros_like(wavelength_grid)

                for _, row in valid_df.iterrows():
                    key = row["conf_key"]
                    weight = row["boltz_weight"]
                    transitions = transition_tables.get(key, [])

                    if len(transitions) == 0:
                        continue

                    uv_y = build_uv_spectrum(
                        transitions,
                        wavelength_grid,
                        broadening_mode=broadening_mode,
                        sigma_nm=sigma_nm,
                        halfwidth_ev=halfwidth_ev
                    )
                    ecd_y = build_ecd_spectrum(
                        transitions,
                        wavelength_grid,
                        broadening_mode=broadening_mode,
                        sigma_nm=sigma_nm,
                        halfwidth_ev=halfwidth_ev
                    )

                    individual_uv_spectra[key] = uv_y
                    individual_ecd_spectra[key] = ecd_y

                    averaged_uv_spectrum += weight * uv_y
                    averaged_ecd_spectrum += weight * ecd_y

                st.subheader("7. 抽出された TD-DFT 遷移")
                for key in df["conf_key"]:
                    with st.expander(f"{key} の遷移を表示"):
                        transitions = transition_tables.get(key, [])
                        if transitions:
                            st.dataframe(pd.DataFrame(transitions))
                        else:
                            st.warning("遷移情報または rotatory strength を抽出できませんでした。")

                show_individual = st.checkbox("各配座スペクトルも表示する", value=True)

                st.subheader("8. UV スペクトル")
                fig_uv, ax_uv = plt.subplots(figsize=(8, 5))

                if show_individual:
                    for key, y in individual_uv_spectra.items():
                        ax_uv.plot(wavelength_grid, y, label=key)

                ax_uv.plot(wavelength_grid, averaged_uv_spectrum, linewidth=2.5, label="Boltzmann-averaged UV")
                ax_uv.set_xlabel("Wavelength (nm)")
                ax_uv.set_ylabel("UV intensity (arb. units)")
                ax_uv.set_title("UV Spectrum")
                ax_uv.legend()

                st.pyplot(fig_uv)

                st.subheader("9. ECD スペクトル")
                fig_ecd, ax_ecd = plt.subplots(figsize=(8, 5))

                if show_individual:
                    for key, y in individual_ecd_spectra.items():
                        ax_ecd.plot(wavelength_grid, y, label=key)

                ax_ecd.plot(wavelength_grid, averaged_ecd_spectrum, linewidth=2.5, label="Boltzmann-averaged ECD")
                ax_ecd.axhline(0, linewidth=1)
                ax_ecd.set_xlabel("Wavelength (nm)")
                ax_ecd.set_ylabel("ECD intensity (arb. units)")
                ax_ecd.set_title("ECD Spectrum")
                ax_ecd.legend()

                st.pyplot(fig_ecd)

                export_df = pd.DataFrame({
                    "wavelength_nm": wavelength_grid,
                    "uv_avg": averaged_uv_spectrum,
                    "ecd_avg": averaged_ecd_spectrum,
                })

                for key, y in individual_uv_spectra.items():
                    export_df[f"uv_{key}"] = y

                for key, y in individual_ecd_spectra.items():
                    export_df[f"ecd_{key}"] = y

                csv_data = export_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "UV / ECD スペクトルCSVをダウンロード",
                    data=csv_data,
                    file_name="uv_ecd_boltzmann_averaged.csv",
                    mime="text/csv"
                )

else:
    st.info("opt/optfreq ファイル群と TD-DFT ファイル群の両方をアップロードしてください。")

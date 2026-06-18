import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# =========================================================
# Page / metadata
# =========================================================
APP_VERSION = "1.2"

DEVELOPER_INFO = {
    "name": "Ken-ichi Nakashima",
    "affiliation_ja": "愛知学院大学 薬学部 薬用資源学講座",
    "affiliation_en": "Aichi-Gakuin University\nSchool of Pharmacy\nLaboratory of Natural Resources",
}

st.set_page_config(page_title="ECD / UV Boltzmann Averaging App", layout="wide")

HARTREE_TO_KCAL = 627.509474
R_KCAL = 0.0019872041  # kcal mol^-1 K^-1
NM_PER_EV = 1239.841984  # lambda(nm) = 1239.841984 / E(eV)


# =========================================================
# UI text
# =========================================================
UI_TEXT = {
    "en": {
        "title": "ECD / UV Boltzmann Averaging App",
        "caption": f"Ver. {APP_VERSION}",
        "description": "Upload opt/optfreq logs and TD-DFT logs separately, match conformers, and perform Boltzmann averaging for UV / ECD spectra.",
        "info": "Files are paired based on a common suffix in the filename. Please make sure each opt/optfreq file and its corresponding TD-DFT file share the same suffix.",
        "settings": "Settings",
        "developer_info": "Developer information",
        "developer_name": "Name",
        "developer_affiliation": "Affiliation",
        "optfreq_header": "1. opt / optfreq files",
        "optfreq_upload": "Select opt or optfreq .log / .out files",
        "tddft_header": "2. TD-DFT files",
        "tddft_upload": "Select TD-DFT .log / .out files",
        "energy_choice": "Select the energy to use for Boltzmann averaging",
        "free_energy": "Free energy",
        "zpe_energy": "Zero-point energy",
        "scf_energy": "SCF energy",
        "temperature": "Temperature (K)",
        "spectrum_settings": "3. Spectrum settings",
        "wl_min": "Minimum wavelength (nm)",
        "wl_max": "Maximum wavelength (nm)",
        "axis_mode": "X-axis specification",
        "point_spacing": "Specify point spacing",
        "n_points": "Specify number of points",
        "point_spacing_nm": "Point spacing (nm)",
        "n_points_label": "Number of points",
        "broadening_mode": "Broadening specification",
        "sigma_nm_mode": "Gaussian broadening width sigma (nm)",
        "halfwidth_ev_mode": "Half-Width (eV)",
        "sigma_nm": "Gaussian broadening width sigma (nm)",
        "halfwidth_ev": "Half-Width (eV)",
        "pairing_header": "4. Pairing results",
        "matched_conformers": "Number of matched conformers",
        "only_opt_warning": "Files found only on the opt/optfreq side: ",
        "only_td_warning": "Files found only on the TD-DFT side: ",
        "no_pairs": "No matchable file pairs were found. Please check the filename convention.",
        "matched_list_header": "5. Matched file list",
        "no_valid_energy": "No conformers were found with the selected energy available.",
        "weights_header": "6. Relative energies and Boltzmann weights",
        "axis_points": "Actual number of x-axis points",
        "transitions_header": "7. Extracted TD-DFT transitions",
        "show_transitions": "Show transitions for {conf_label}",
        "transition_extract_fail": "Failed to extract transition information or rotatory strengths.",
        "show_individual": "Show individual conformer spectra",
        "uv_header": "8. UV spectrum",
        "uv_ylabel": "UV intensity (arb. units)",
        "uv_title": "UV Spectrum",
        "uv_avg_label": "Boltzmann-averaged UV",
        "ecd_header": "9. ECD spectrum",
        "ecd_ylabel": "ECD intensity (arb. units)",
        "ecd_title": "ECD Spectrum",
        "ecd_avg_label": "Boltzmann-averaged ECD",
        "download_csv": "Download UV / ECD spectra CSV",
        "download_filename": "uv_ecd_boltzmann_averaged.csv",
        "need_uploads": "Please upload both opt/optfreq files and TD-DFT files.",
        "output_prefix_info": "Download filename uses the common prefix of uploaded input files.",
    },
    "ja": {
        "title": "ECD / UV Boltzmann Averaging App",
        "caption": f"Ver. {APP_VERSION}",
        "description": "opt / optfreq ログと TD-DFT ログを別々にアップロードし、配座を対応付けて UV / ECD スペクトルの Boltzmann 平均を計算します。",
        "info": "ファイルはファイル名の共通接尾辞に基づいて対応付けられます。opt / optfreq ファイルと対応する TD-DFT ファイルで、同じ接尾辞を持つようにしてください。",
        "settings": "設定",
        "developer_info": "開発者情報",
        "developer_name": "氏名",
        "developer_affiliation": "所属",
        "optfreq_header": "1. opt / optfreq ファイル",
        "optfreq_upload": "opt または optfreq の .log / .out ファイルを選択",
        "tddft_header": "2. TD-DFT ファイル",
        "tddft_upload": "TD-DFT の .log / .out ファイルを選択",
        "energy_choice": "Boltzmann 平均に用いるエネルギーを選択",
        "free_energy": "自由エネルギー",
        "zpe_energy": "ゼロ点エネルギー",
        "scf_energy": "SCF エネルギー",
        "temperature": "温度 (K)",
        "spectrum_settings": "3. スペクトル設定",
        "wl_min": "最小波長 (nm)",
        "wl_max": "最大波長 (nm)",
        "axis_mode": "X軸の指定方法",
        "point_spacing": "点間隔を指定",
        "n_points": "点数を指定",
        "point_spacing_nm": "点間隔 (nm)",
        "n_points_label": "点数",
        "broadening_mode": "ブロードニング指定",
        "sigma_nm_mode": "ガウス幅 sigma (nm)",
        "halfwidth_ev_mode": "Half-Width (eV)",
        "sigma_nm": "ガウス幅 sigma (nm)",
        "halfwidth_ev": "Half-Width (eV)",
        "pairing_header": "4. 対応付け結果",
        "matched_conformers": "対応付けられた配座数",
        "only_opt_warning": "opt / optfreq 側のみに存在するファイル: ",
        "only_td_warning": "TD-DFT 側のみに存在するファイル: ",
        "no_pairs": "対応付け可能なファイルペアが見つかりませんでした。ファイル名規則を確認してください。",
        "matched_list_header": "5. 対応付けられたファイル一覧",
        "no_valid_energy": "選択したエネルギーが利用可能な配座が見つかりませんでした。",
        "weights_header": "6. 相対エネルギーと Boltzmann 存在比",
        "axis_points": "実際のX軸点数",
        "transitions_header": "7. 抽出された TD-DFT 遷移",
        "show_transitions": "{conf_label} の遷移を表示",
        "transition_extract_fail": "遷移情報または rotatory strength の抽出に失敗しました。",
        "show_individual": "各配座のスペクトルを表示",
        "uv_header": "8. UV スペクトル",
        "uv_ylabel": "UV 強度 (arb. units)",
        "uv_title": "UV Spectrum",
        "uv_avg_label": "Boltzmann 平均 UV",
        "ecd_header": "9. ECD スペクトル",
        "ecd_ylabel": "ECD 強度 (arb. units)",
        "ecd_title": "ECD Spectrum",
        "ecd_avg_label": "Boltzmann 平均 ECD",
        "download_csv": "UV / ECD スペクトル CSV をダウンロード",
        "download_filename": "uv_ecd_boltzmann_averaged.csv",
        "need_uploads": "opt / optfreq ファイルと TD-DFT ファイルの両方をアップロードしてください。",
        "output_prefix_info": "ダウンロードファイル名には入力ファイル名の共通接頭辞を使用します。",
    },
}


# =========================================================
# Session state
# =========================================================
if "ui_language" not in st.session_state:
    st.session_state["ui_language"] = "English"


def current_language():
    return "ja" if st.session_state.get("ui_language", "English") == "日本語" else "en"


def t(key: str, **kwargs):
    text = UI_TEXT[current_language()].get(key, key)
    return text.format(**kwargs) if kwargs else text


# =========================================================
# Filename helpers
# =========================================================
def sanitize_filename_part(text):
    text = Path(text).stem
    text = re.sub(r"[^\w\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.-")
    return text or "output"


def longest_common_prefix(strings):
    if not strings:
        return ""

    prefix = strings[0]
    for s in strings[1:]:
        i = 0
        max_len = min(len(prefix), len(s))
        while i < max_len and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


def clean_common_prefix(prefix):
    prefix = prefix.strip("_.- ")
    prefix = re.sub(r"[_\-.]+$", "", prefix)
    return prefix


def build_output_prefix_from_inputs(optfreq_files, tddft_files, min_prefix_len=3, fallback="output"):
    names = []
    if optfreq_files:
        names.extend([sanitize_filename_part(f.name) for f in optfreq_files])
    if tddft_files:
        names.extend([sanitize_filename_part(f.name) for f in tddft_files])

    if not names:
        return fallback

    if len(names) == 1:
        return names[0]

    prefix = longest_common_prefix(names)
    prefix = clean_common_prefix(prefix)

    if len(prefix) >= min_prefix_len:
        return prefix

    return fallback


# =========================================================
# Header
# =========================================================
with st.sidebar:
    selected_language = st.selectbox(
        "Language / 言語",
        options=["English", "日本語"],
        index=0 if st.session_state["ui_language"] == "English" else 1,
    )
    st.session_state["ui_language"] = selected_language

st.title(t("title"))
st.caption(t("caption"))
st.write(t("description"))
st.info(t("info"))

st.sidebar.header(t("settings"))
with st.sidebar.expander(t("developer_info"), expanded=False):
    if current_language() == "ja":
        affiliation_text = DEVELOPER_INFO["affiliation_ja"]
    else:
        affiliation_text = DEVELOPER_INFO["affiliation_en"]

    st.markdown(
        f"""
**{t("developer_name")}**  
{DEVELOPER_INFO["name"]}

**{t("developer_affiliation")}**  
{affiliation_text}
"""
    )


# --------------------------------------------------
# Helper: normalize filenames for pairing
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
# Energy extraction (opt/optfreq)
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
# TD-DFT transition extraction
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
# Rotatory strength extraction
# --------------------------------------------------
def extract_rotatory_strengths(text, mode="length"):
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
        row["rot_strength"] = rot_strengths_len[i]
        row["rot_strength_length"] = rot_strengths_len[i]
        row["rot_strength_velocity"] = (
            rot_strengths_vel[i] if i < len(rot_strengths_vel) else None
        )
        transitions.append(row)

    return transitions


# --------------------------------------------------
# Math helpers
# --------------------------------------------------
def safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return 0.0


def gaussian_broadening(x, center, height, sigma):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def halfwidth_to_sigma(value):
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
# Spectrum builders
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
    e_min = NM_PER_EV / np.max(wavelength_grid_nm)
    e_max = NM_PER_EV / np.min(wavelength_grid_nm)

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
st.subheader(t("optfreq_header"))
optfreq_files = st.file_uploader(
    t("optfreq_upload"),
    type=["log", "out"],
    accept_multiple_files=True,
    key="optfreq_files"
)

st.subheader(t("tddft_header"))
tddft_files = st.file_uploader(
    t("tddft_upload"),
    type=["log", "out"],
    accept_multiple_files=True,
    key="tddft_files"
)

output_prefix = build_output_prefix_from_inputs(optfreq_files, tddft_files)
if optfreq_files or tddft_files:
    st.caption(t("output_prefix_info"))

energy_choice = st.selectbox(
    t("energy_choice"),
    options=["free_energy", "zpe_energy", "scf_energy"],
    format_func=lambda x: {
        "free_energy": t("free_energy"),
        "zpe_energy": t("zpe_energy"),
        "scf_energy": t("scf_energy"),
    }[x]
)

temperature = st.number_input(
    t("temperature"),
    min_value=1.0,
    value=298.15,
    step=1.0
)

st.subheader(t("spectrum_settings"))
wl_min = st.number_input(t("wl_min"), value=150.0)
wl_max = st.number_input(t("wl_max"), value=400.0)

axis_mode = st.radio(
    t("axis_mode"),
    options=["point_spacing", "n_points"],
    format_func=lambda x: {
        "point_spacing": t("point_spacing"),
        "n_points": t("n_points"),
    }[x]
)

if axis_mode == "point_spacing":
    point_spacing_nm = st.number_input(
        t("point_spacing_nm"),
        min_value=0.001,
        value=0.2,
        step=0.1,
        format="%.4f"
    )
    n_points = None
else:
    n_points = st.number_input(
        t("n_points_label"),
        min_value=2,
        value=2000,
        step=100
    )
    point_spacing_nm = None

broadening_mode = st.radio(
    t("broadening_mode"),
    options=["sigma_nm", "halfwidth_ev"],
    format_func=lambda x: {
        "sigma_nm": t("sigma_nm_mode"),
        "halfwidth_ev": t("halfwidth_ev_mode"),
    }[x]
)

if broadening_mode == "sigma_nm":
    sigma_nm = st.number_input(
        t("sigma_nm"),
        min_value=0.001,
        value=10.0,
        step=0.5
    )
    halfwidth_ev = None
else:
    halfwidth_ev = st.number_input(
        t("halfwidth_ev"),
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

    st.subheader(t("pairing_header"))
    st.write(f"{t('matched_conformers')}: {len(pairs)}")

    if only_energy_keys:
        st.warning(t("only_opt_warning") + ", ".join(only_energy_keys))

    if only_tddft_keys:
        st.warning(t("only_td_warning") + ", ".join(only_tddft_keys))

    if len(pairs) == 0:
        st.error(t("no_pairs"))
    else:
        records = []
        transition_tables = {}

        for i, pair in enumerate(pairs, start=1):
            opt_name = pair["opt_name"]
            td_name = pair["td_name"]

            e = optfreq_data[opt_name]
            td = tddft_data[td_name]

            conf_label = f"pair_{i:02d}"

            records.append({
                "conf_key": conf_label,
                "optfreq_file": e["optfreq_file"],
                "tddft_file": td["tddft_file"],
                "suffix_token_score": pair["token_score"],
                "suffix_char_score": pair["char_score"],
                "scf_energy": e["scf_energy"],
                "zpe_energy": e["zpe_energy"],
                "free_energy": e["free_energy"],
                "n_transitions": td["n_transitions"],
            })

            transition_tables[conf_label] = td["transitions"]

        df = pd.DataFrame(records)

        st.subheader(t("matched_list_header"))
        st.dataframe(df)

        valid_df = df[df[energy_choice].notna()].copy()

        if valid_df.empty:
            st.error(t("no_valid_energy"))
        else:
            e_min = valid_df[energy_choice].min()
            valid_df["delta_E_hartree"] = valid_df[energy_choice] - e_min
            valid_df["delta_E_kcal_mol"] = valid_df["delta_E_hartree"] * HARTREE_TO_KCAL
            valid_df["boltz_factor"] = valid_df["delta_E_kcal_mol"].apply(
                lambda x: safe_exp(-x / (R_KCAL * temperature))
            )

            factor_sum = valid_df["boltz_factor"].sum()

            if factor_sum == 0:
                st.error("The sum of Boltzmann factors became zero.")
            else:
                valid_df["boltz_weight"] = valid_df["boltz_factor"] / factor_sum
                valid_df = valid_df.sort_values(by=energy_choice).reset_index(drop=True)

                st.subheader(t("weights_header"))
                st.dataframe(valid_df)

                wavelength_grid = make_wavelength_grid(
                    wl_min=wl_min,
                    wl_max=wl_max,
                    axis_mode=axis_mode,
                    point_spacing_nm=point_spacing_nm,
                    n_points=n_points
                )

                st.write(f"{t('axis_points')}: {len(wavelength_grid)}")

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

                st.subheader(t("transitions_header"))
                for key in df["conf_key"]:
                    with st.expander(t("show_transitions", conf_label=key)):
                        transitions = transition_tables.get(key, [])
                        if transitions:
                            st.dataframe(pd.DataFrame(transitions))
                        else:
                            st.warning(t("transition_extract_fail"))

                show_individual = st.checkbox(t("show_individual"), value=False)

                st.subheader(t("uv_header"))
                fig_uv, ax_uv = plt.subplots(figsize=(8, 5))

                if show_individual:
                    for key, y in individual_uv_spectra.items():
                        ax_uv.plot(wavelength_grid, y, label=key)

                ax_uv.plot(wavelength_grid, averaged_uv_spectrum, linewidth=2.5, label=t("uv_avg_label"))
                ax_uv.set_xlabel("Wavelength (nm)")
                ax_uv.set_ylabel(t("uv_ylabel"))
                ax_uv.set_title(t("uv_title"))
                ax_uv.legend()

                st.pyplot(fig_uv)

                st.subheader(t("ecd_header"))
                fig_ecd, ax_ecd = plt.subplots(figsize=(8, 5))

                if show_individual:
                    for key, y in individual_ecd_spectra.items():
                        ax_ecd.plot(wavelength_grid, y, label=key)

                ax_ecd.plot(wavelength_grid, averaged_ecd_spectrum, linewidth=2.5, label=t("ecd_avg_label"))
                ax_ecd.axhline(0, linewidth=1)
                ax_ecd.set_xlabel("Wavelength (nm)")
                ax_ecd.set_ylabel(t("ecd_ylabel"))
                ax_ecd.set_title(t("ecd_title"))
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
                    t("download_csv"),
                    data=csv_data,
                    file_name=f"{output_prefix}_{t('download_filename')}",
                    mime="text/csv"
                )

else:
    st.info(t("need_uploads"))

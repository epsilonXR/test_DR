from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score



def get_project_root() -> Path:
    """
    tools/tools.py 获得项目根目录的路径。
    """
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> dict:
    """
    读取path.dev.toml
    """
    if config_path is None:
        config_path = get_project_root() / "path.dev.toml"
    else:
        config_path = Path(config_path)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    return config


def get_aligned_paths(model_name: str, config: dict | None = None) -> tuple[Path, Path]:
    """
    从模型特征空间名称构造aligned_drvalid的npz和metadata路径。
    例:
    model_name = "FVs_medsiglip"
    -> aligned_drvalid_FVs_medsiglip.npz
    -> aligned_drvalid_FVs_medsiglip_metadata.csv
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    aligned_dir = root / config["paths"]["aligned_dir"]

    npz_path = aligned_dir / f"aligned_drvalid_{model_name}.npz"
    meta_path = aligned_dir / f"aligned_drvalid_{model_name}_metadata.csv"

    if not npz_path.exists():
        raise FileNotFoundError(f"npz file not found: {npz_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata csv not found: {meta_path}")

    return npz_path, meta_path


def load_aligned_features(model_name: str, config: dict | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """
    从aligned_features_drvalid中加载特征向量X和metadata。
    """
    npz_path, meta_path = get_aligned_paths(model_name, config)

    data = np.load(npz_path, allow_pickle=True)
    if "X" not in data:
        raise KeyError(f"'X' not found in {npz_path}. Available keys: {list(data.keys())}")

    X = data["X"]
    meta_df = pd.read_csv(meta_path)

    if len(X) != len(meta_df):
        raise ValueError(
            f"X and metadata length mismatch: X={len(X)}, meta={len(meta_df)}"
        )

    if np.isnan(X).any():
        raise ValueError("X contains NaN values.")

    return X, meta_df


def run_pca(
    X: np.ndarray,
    n_components: int = 50,
    standardize: bool = True,
) -> dict:
    """
    对特征矩阵X进行PCA处理,并返回PCA结果和相关信息。
    """
    if standardize:
        scaler = StandardScaler()
        X_input = scaler.fit_transform(X)
    else:
        scaler = None
        X_input = X

    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X_input)

    return {
        "Z": Z,
        "pca": pca,
        "scaler": scaler,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    }


def make_pca_df(Z: np.ndarray) -> pd.DataFrame:
    """
    返回一个DataFrame,其中包含PCA降维后的特征Z。
    """
    n_components = Z.shape[1]
    pc_cols = [f"PC{i+1}" for i in range(n_components)]

    pca_df = pd.DataFrame(Z, columns=pc_cols)

    return pca_df


def pca_for_model(
    model_name: str,
    config: dict | None = None,
    n_components: int | None = None,
    standardize: bool | None = None,
) -> dict:
    """
    通过指定model_name,执行特征读取 → 标准化 → PCA → pca_df创建的主函数。
    """
    if config is None:
        config = load_config()

    if n_components is None:
        n_components = config["pca"].get("n_components", 50)

    if standardize is None:
        standardize = config["pca"].get("standardize", True)

    X, meta_df = load_aligned_features(model_name, config)

    pca_result = run_pca(
        X,
        n_components=n_components,
        standardize=standardize,
    )

    pca_df = make_pca_df(pca_result["Z"])

    return {
        "model_name": model_name,
        "X": X,
        "meta_df": meta_df,
        "pca_df": pca_df,
        "pca": pca_result["pca"],
        "scaler": pca_result["scaler"],
        "explained_variance_ratio": pca_result["explained_variance_ratio"],
        "cumulative_explained_variance": pca_result["cumulative_explained_variance"],
    }

def _normalize_pc_list(pc_list, n_available: int) -> list[int]:
    """
    PC番号を1始まりで受け取り、0始まりindexに変換する。

    例:
    pc_list=[1, 3, 5] -> [0, 2, 4]
    """
    if pc_list is None:
        return list(range(n_available))

    if isinstance(pc_list, int):
        pc_list = [pc_list]

    indices = []
    for pc in pc_list:
        if pc < 1 or pc > n_available:
            raise ValueError(
                f"Invalid PC number: PC{pc}. "
                f"Available range is PC1 to PC{n_available}."
            )
        indices.append(pc - 1)

    return indices


def get_pc_component_df(pca, feature_names: list[str] | None = None, n_pc: int | None = None) -> pd.DataFrame:
    """
    主成分方向ベクトルをDataFrameとして返す。

    pca.components_ の形:
        (n_components, n_features)

    各行が PC1, PC2, ...
    各列が元の特徴次元。
    """
    components = pca.components_

    if n_pc is not None:
        components = components[:n_pc]

    n_features = components.shape[1]

    if feature_names is None:
        feature_names = [f"FV{i}" for i in range(n_features)]

    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length mismatch: "
            f"{len(feature_names)} != {n_features}"
        )

    pc_names = [f"PC{i+1}" for i in range(components.shape[0])]

    return pd.DataFrame(
        components,
        index=pc_names,
        columns=feature_names,
    )


def compute_pc_scores(
    X: np.ndarray,
    pca,
    scaler=None,
    pc_list: list[int] | int | None = None,
    as_df: bool = True,
) -> pd.DataFrame | np.ndarray:
    """
    特徴ベクトル X を PCA 空間に射影し、PC score を返す。

    これは pca.transform(X_std) に相当する。
    """
    X = np.asarray(X)

    if scaler is not None:
        X_input = scaler.transform(X)
    else:
        X_input = X

    Z = pca.transform(X_input)

    indices = _normalize_pc_list(pc_list, Z.shape[1])
    Z_sel = Z[:, indices]

    if not as_df:
        return Z_sel

    pc_names = [f"PC{i+1}" for i in indices]

    return pd.DataFrame(Z_sel, columns=pc_names)


def project_onto_pc_subspace(
    X: np.ndarray,
    pca,
    scaler=None,
    pc_list: list[int] | int | None = None,
) -> np.ndarray:
    """
    指定したPC部分空間への射影ベクトルを返す。

    戻り値は PCA 入力空間上の centered projection。
    つまり、標準化してPCAした場合は「標準化済み特徴空間」での射影。

    数式:
        projection = sum_k z_k v_k
    """
    X = np.asarray(X)

    if scaler is not None:
        X_input = scaler.transform(X)
    else:
        X_input = X

    Z = pca.transform(X_input)

    indices = _normalize_pc_list(pc_list, Z.shape[1])

    Z_sel = Z[:, indices]
    V_sel = pca.components_[indices]

    X_proj_centered = Z_sel @ V_sel

    return X_proj_centered


def keep_only_pcs(
    X: np.ndarray,
    pca,
    scaler=None,
    pc_list: list[int] | int | None = None,
    original_scale: bool = False,
) -> np.ndarray:
    """
    指定したPC成分だけを残して、特徴空間に戻す。

    例:
        keep_only_pcs(X, pca, scaler, pc_list=[1])
        -> PC1成分だけで再構成した特徴ベクトル

    original_scale=True の場合:
        StandardScaler の inverse_transform で元の特徴スケールに戻す。
    """
    X_proj_centered = project_onto_pc_subspace(
        X=X,
        pca=pca,
        scaler=scaler,
        pc_list=pc_list,
    )

    X_keep_input = pca.mean_ + X_proj_centered

    if original_scale:
        if scaler is None:
            raise ValueError("original_scale=True requires scaler.")
        return scaler.inverse_transform(X_keep_input)

    return X_keep_input


def remove_pcs_from_features(
    X: np.ndarray,
    pca,
    scaler=None,
    pc_list: list[int] | int | None = None,
    original_scale: bool = False,
) -> np.ndarray:
    """
    指定したPC成分を特徴ベクトルから除去する。

    数式:
        X_removed = X - projection_to_selected_PCs

    標準化してPCAした場合、
    除去は標準化済み特徴空間で行われる。

    original_scale=True の場合:
        除去後の特徴を元の特徴スケールに戻す。
    """
    X = np.asarray(X)

    if scaler is not None:
        X_input = scaler.transform(X)
    else:
        X_input = X

    X_proj_centered = project_onto_pc_subspace(
        X=X,
        pca=pca,
        scaler=scaler,
        pc_list=pc_list,
    )

    X_removed_input = X_input - X_proj_centered

    if original_scale:
        if scaler is None:
            raise ValueError("original_scale=True requires scaler.")
        return scaler.inverse_transform(X_removed_input)

    return X_removed_input


def decompose_features_by_pcs(
    X: np.ndarray,
    pca,
    scaler=None,
    pc_list: list[int] | int | None = None,
    original_scale: bool = False,
) -> dict:
    """
    指定PCについて、score / projection / keep / remove をまとめて返す。
    """
    scores_df = compute_pc_scores(
        X=X,
        pca=pca,
        scaler=scaler,
        pc_list=pc_list,
        as_df=True,
    )

    projection = project_onto_pc_subspace(
        X=X,
        pca=pca,
        scaler=scaler,
        pc_list=pc_list,
    )

    kept = keep_only_pcs(
        X=X,
        pca=pca,
        scaler=scaler,
        pc_list=pc_list,
        original_scale=original_scale,
    )

    removed = remove_pcs_from_features(
        X=X,
        pca=pca,
        scaler=scaler,
        pc_list=pc_list,
        original_scale=original_scale,
    )

    return {
        "scores_df": scores_df,
        "projection": projection,
        "kept_features": kept,
        "removed_features": removed,
    }




def get_pc_columns(pca_df: pd.DataFrame, n_pc: int | None = None) -> list[str]:
    """
    从 pca_df 中获取 PC1, PC2, ... 列名,并按照主成分编号排序。

    参数：
        pca_df:
            只包含主成分得分的 DataFrame。

        n_pc:
            使用前多少个主成分。如果为 None,则使用所有 PC 列。
    """
    pc_cols = [c for c in pca_df.columns if c.startswith("PC")]
    pc_cols = sorted(pc_cols, key=lambda x: int(x.replace("PC", "")))

    if n_pc is not None:
        pc_cols = pc_cols[:n_pc]

    return pc_cols


def safe_binary_auc(y, score):
    """
    安全计算二分类 AUC。

    y:
        二值标签,必须是 0 / 1。

    score:
        连续得分,例如某个 PC 的主成分得分。

    返回：
        auc_raw:
            原始 AUC,不做方向修正。
    """
    y = np.asarray(y)
    score = np.asarray(score)

    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(y, score)


def compute_pc_auc_table(
    pca_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    auc_cols: list[str],
    n_pc: int = 50,
    positive_label_map: dict | None = None,
) -> pd.DataFrame:
    """
    计算每个 PC 与二值因子的 AUC 关系。

    适用对象：
        - eye
        - view_no
        - Overall quality
        - 其他二值变量

    AUC 组的统一强度标准：
        score = 2 * abs(AUC_raw - 0.5)

    参数：
        pca_df:
            只包含 PC1, PC2, ... 的主成分得分表。

        meta_df:
            metadata 表。

        auc_cols:
            需要用 AUC 分析的因子列名。

        n_pc:
            使用前多少个 PC。

        positive_label_map:
            指定每个因子的正类标签。
            例如：
                {
                    "Overall quality": 1,
                    "eye": "right"
                }
            如果不指定,则默认把排序后的第二个类别作为正类。
    """
    if positive_label_map is None:
        positive_label_map = {}

    pc_cols = get_pc_columns(pca_df, n_pc=n_pc)
    rows = []

    for factor in auc_cols:
        if factor not in meta_df.columns:
            raise KeyError(f"meta_df 中不存在列: {factor}")

        factor_values = meta_df[factor].dropna().unique()

        if len(factor_values) != 2:
            raise ValueError(
                f"{factor} 不是二值变量,不能直接用 binary AUC。"
                f"当前唯一值为: {sorted(factor_values)}"
            )

        if factor in positive_label_map:
            positive_label = positive_label_map[factor]
        else:
            positive_label = sorted(factor_values)[1]

        negative_label = [v for v in sorted(factor_values) if v != positive_label][0]

        for pc in pc_cols:
            tmp = pd.DataFrame({
                "pc_score": pca_df[pc],
                "factor": meta_df[factor],
            }).dropna()

            y_binary = (tmp["factor"] == positive_label).astype(int).to_numpy()
            pc_score = tmp["pc_score"].to_numpy()

            auc_raw = safe_binary_auc(y_binary, pc_score)

            if pd.isna(auc_raw):
                auc_signed_score = np.nan
                score = np.nan
            else:
                auc_signed_score = 2 * (auc_raw - 0.5)
                score = 2 * abs(auc_raw - 0.5)

            rows.append({
                "PC": pc,
                "factor": factor,
                "method": "auc",
                "raw_value": auc_raw,
                "score": score,
                "signed_score": auc_signed_score,
                "auc_raw": auc_raw,
                "auc_signed_score": auc_signed_score,
                "spearman_rho": np.nan,
                "p_value": np.nan,
                "negative_label": negative_label,
                "positive_label": positive_label,
                "n": len(tmp),
                "n_negative": int((tmp["factor"] == negative_label).sum()),
                "n_positive": int((tmp["factor"] == positive_label).sum()),
                "mean_PC_negative": tmp.loc[tmp["factor"] == negative_label, "pc_score"].mean(),
                "mean_PC_positive": tmp.loc[tmp["factor"] == positive_label, "pc_score"].mean(),
            })

    return pd.DataFrame(rows)


def compute_pc_spearman_table(
    pca_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    spearman_cols: list[str],
    n_pc: int = 50,
) -> pd.DataFrame:
    """
    计算每个 PC 与有序因子的 Spearman 相关。

    适用对象：
        - patient_DR_Level
        - eye_DR_Level
        - Clarity
        - Field definition
        - Artifact
        - 其他有序评分变量

    Spearman 组的统一强度标准：
        score = abs(spearman_rho)

    参数：
        pca_df:
            只包含 PC1, PC2, ... 的主成分得分表。

        meta_df:
            metadata 表。

        spearman_cols:
            需要用 Spearman 分析的因子列名。

        n_pc:
            使用前多少个 PC。
    """
    pc_cols = get_pc_columns(pca_df, n_pc=n_pc)
    rows = []

    for factor in spearman_cols:
        if factor not in meta_df.columns:
            raise KeyError(f"meta_df 中不存在列: {factor}")

        for pc in pc_cols:
            tmp = pd.DataFrame({
                "pc_score": pca_df[pc],
                "factor": meta_df[factor],
            }).dropna()

            if tmp["pc_score"].nunique() < 2 or tmp["factor"].nunique() < 2:
                rho = np.nan
                p_value = np.nan
            else:
                rho, p_value = spearmanr(tmp["pc_score"], tmp["factor"])

            rows.append({
                "PC": pc,
                "factor": factor,
                "method": "spearman",
                "raw_value": rho,
                "score": abs(rho) if pd.notna(rho) else np.nan,
                "signed_score": rho,
                "auc_raw": np.nan,
                "auc_signed_score": np.nan,
                "spearman_rho": rho,
                "p_value": p_value,
                "negative_label": np.nan,
                "positive_label": np.nan,
                "n": len(tmp),
                "n_negative": np.nan,
                "n_positive": np.nan,
                "mean_PC_negative": np.nan,
                "mean_PC_positive": np.nan,
            })

    return pd.DataFrame(rows)


def compute_pc_factor_score_table(
    pca_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    auc_cols: list[str] | None = None,
    spearman_cols: list[str] | None = None,
    n_pc: int = 50,
    positive_label_map: dict | None = None,
) -> pd.DataFrame:
    """
    统一计算 PC × factor 的关系强度表。

    AUC 因子：
        score = 2 * abs(AUC_raw - 0.5)

    Spearman 因子：
        score = abs(spearman_rho)

    返回：
        long format 的结果表。
        后续可以直接转成 heatmap matrix。
    """
    if auc_cols is None:
        auc_cols = []

    if spearman_cols is None:
        spearman_cols = []

    result_list = []

    if len(auc_cols) > 0:
        auc_df = compute_pc_auc_table(
            pca_df=pca_df,
            meta_df=meta_df,
            auc_cols=auc_cols,
            n_pc=n_pc,
            positive_label_map=positive_label_map,
        )
        result_list.append(auc_df)

    if len(spearman_cols) > 0:
        spearman_df = compute_pc_spearman_table(
            pca_df=pca_df,
            meta_df=meta_df,
            spearman_cols=spearman_cols,
            n_pc=n_pc,
        )
        result_list.append(spearman_df)

    if len(result_list) == 0:
        return pd.DataFrame()

    result_df = pd.concat(result_list, axis=0, ignore_index=True)

    return result_df


def make_pc_factor_matrix(
    result_df: pd.DataFrame,
    value_col: str = "score",
) -> pd.DataFrame:
    """
    将 PC × factor 的长表转换成热力图矩阵。

    默认 value_col = "score"。

    也可以指定：
        - "raw_value"
        - "signed_score"
        - "auc_raw"
        - "auc_signed_score"
        - "spearman_rho"
    """
    matrix_df = result_df.pivot(
        index="PC",
        columns="factor",
        values=value_col,
    )

    matrix_df = matrix_df.loc[
        sorted(matrix_df.index, key=lambda x: int(x.replace("PC", "")))
    ]

    return matrix_df


def plot_pc_factor_heatmap(
    matrix_df: pd.DataFrame,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 14),
    annot: bool = False,
    vmin: float | None = 0,
    vmax: float | None = 1,
    cmap: str = "Blues",
):
    """
    绘制 PC × factor 的热力图。

    默认用于画 score 强度图：
        AUC: 2 * abs(AUC_raw - 0.5)
        Spearman: abs(rho)
    """
    plt.figure(figsize=figsize)

    sns.heatmap(
        matrix_df,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "score"},
    )

    if title is not None:
        plt.title(title)

    plt.xlabel("Factor")
    plt.ylabel("Principal Component")
    plt.tight_layout()
    plt.show()

import re

def infer_view_no_from_image_path(image_path):
    """
    从 image_path 中推断 view_no。

    例：
        \\regular-fundus-training\\5\\5_l1.jpg -> view_no = 1
        \\regular-fundus-training\\5\\5_l2.jpg -> view_no = 2
        \\regular-fundus-training\\5\\5_r1.jpg -> view_no = 1
        \\regular-fundus-training\\5\\5_r2.jpg -> view_no = 2
    """
    if pd.isna(image_path):
        return np.nan

    text = str(image_path).strip()

    # 兼容 Windows 路径和 Linux 路径,只取最后的文件名
    file_name = re.split(r"[\\/]", text)[-1]

    # 去掉扩展名,例如 5_l1.jpg -> 5_l1
    stem = re.sub(r"\.[^.]+$", "", file_name)

    # 匹配结尾的 l1 / l2 / r1 / r2
    match = re.search(r"(?i)([lr])([12])$", stem)

    if match is None:
        return np.nan

    view_no = int(match.group(2))
    return view_no


def add_view_no_column(
    meta_df: pd.DataFrame,
    image_path_col: str = "image_path",
    view_col: str = "view_no",
    overwrite: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """
    根据 image_path 添加 view_no 列。

    参数：
        meta_df:
            metadata 表。

        image_path_col:
            保存图像路径的列名。

        view_col:
            新增的 view_no 列名。

        overwrite:
            如果已经存在 view_no,是否覆盖。

        strict:
            如果存在无法解析的路径,是否直接报错。
    """
    meta_df = meta_df.copy()

    if image_path_col not in meta_df.columns:
        raise KeyError(f"meta_df 中不存在列: {image_path_col}")

    if view_col in meta_df.columns and not overwrite:
        return meta_df

    meta_df[view_col] = meta_df[image_path_col].apply(infer_view_no_from_image_path)
    meta_df[view_col] = meta_df[view_col].astype("Int64")

    failed_mask = meta_df[view_col].isna()

    if strict and failed_mask.any():
        examples = meta_df.loc[failed_mask, image_path_col].head(10)
        raise ValueError(
            f"有 {failed_mask.sum()} 行无法从 image_path 解析 view_no。\n"
            f"前几个例子：\n{examples}"
        )

    return meta_df

def select_pcs_to_remove(
    score_df: pd.DataFrame,
    disease_cols: list[str],
    nuisance_cols: list[str],
    nuisance_threshold: float = 0.30,
    disease_threshold: float = 0.20,
    margin_threshold: float = 0.10,
    max_remove: int | None = None,
) -> dict:
    """
    根据 PC × factor score 表，自动选择需要去除的主成分。

    判定标准：
        1. nuisance_score >= nuisance_threshold
        2. disease_score <= disease_threshold
        3. nuisance_score - disease_score >= margin_threshold

    其中：
        disease_score:
            该 PC 与病变因子之间的最大关系强度。
            例如 patient_DR_Level / eye_DR_Level 的 max score。

        nuisance_score:
            该 PC 与非病变因子之间的最大关系强度。
            例如 Overall quality / eye / view_no / Clarity / Field definition / Artifact 的 max score。

    参数：
        score_df:
            compute_pc_factor_score_table() 的输出结果。

        disease_cols:
            病变相关因子列名。
            例如 ["patient_DR_Level", "eye_DR_Level"]

        nuisance_cols:
            非病变相关因子列名。
            例如 ["Overall quality", "eye", "view_no", "Clarity", "Field definition", "Artifact"]

        nuisance_threshold:
            非病变得分阈值。
            默认 0.30，表示该 PC 对某个非病变因子有比较明显的解释能力。

        disease_threshold:
            病变得分上限。
            默认 0.20，表示该 PC 与病变严重度关系不能太强。

        margin_threshold:
            非病变得分必须比病变得分高多少。
            默认 0.10，用来避免把病变 PC 误删。

        max_remove:
            最多去除几个 PC。
            如果为 None，则不限制数量。

    返回：
        {
            "remove_pc_list": [需要去除的 PC 编号，整数形式],
            "remove_pc_names": [需要去除的 PC 名称，例如 "PC5"],
            "selection_df": 每个 PC 的判定详情表
        }
    """
    required_cols = {"PC", "factor", "score", "method", "signed_score"}
    missing_cols = required_cols - set(score_df.columns)

    if missing_cols:
        raise KeyError(f"score_df 缺少必要列: {missing_cols}")

    rows = []

    pc_list = sorted(
        score_df["PC"].dropna().unique(),
        key=lambda x: int(str(x).replace("PC", ""))
    )

    for pc in pc_list:
        pc_df = score_df[score_df["PC"] == pc].copy()

        disease_df = pc_df[pc_df["factor"].isin(disease_cols)].copy()
        nuisance_df = pc_df[pc_df["factor"].isin(nuisance_cols)].copy()

        if len(disease_df) == 0:
            disease_score = np.nan
            disease_factor = np.nan
            disease_method = np.nan
            disease_signed_score = np.nan
        else:
            disease_idx = disease_df["score"].idxmax()
            disease_row = disease_df.loc[disease_idx]
            disease_score = disease_row["score"]
            disease_factor = disease_row["factor"]
            disease_method = disease_row["method"]
            disease_signed_score = disease_row["signed_score"]

        if len(nuisance_df) == 0:
            nuisance_score = np.nan
            nuisance_factor = np.nan
            nuisance_method = np.nan
            nuisance_signed_score = np.nan
        else:
            nuisance_idx = nuisance_df["score"].idxmax()
            nuisance_row = nuisance_df.loc[nuisance_idx]
            nuisance_score = nuisance_row["score"]
            nuisance_factor = nuisance_row["factor"]
            nuisance_method = nuisance_row["method"]
            nuisance_signed_score = nuisance_row["signed_score"]

        if pd.isna(disease_score) or pd.isna(nuisance_score):
            margin = np.nan
            remove_flag = False
            reason = "缺少 disease_score 或 nuisance_score"
        else:
            margin = nuisance_score - disease_score

            remove_flag = (
                nuisance_score >= nuisance_threshold
                and disease_score <= disease_threshold
                and margin >= margin_threshold
            )

            if remove_flag:
                reason = (
                    f"非病变因子 {nuisance_factor} 的 score={nuisance_score:.3f} 较高，"
                    f"病变 score={disease_score:.3f} 较低，"
                    f"差值={margin:.3f}"
                )
            else:
                reason = "不满足去除条件"

        rows.append({
            "PC": pc,
            "pc_number": int(str(pc).replace("PC", "")),
            "remove": remove_flag,

            "disease_score": disease_score,
            "disease_factor": disease_factor,
            "disease_method": disease_method,
            "disease_signed_score": disease_signed_score,

            "nuisance_score": nuisance_score,
            "nuisance_factor": nuisance_factor,
            "nuisance_method": nuisance_method,
            "nuisance_signed_score": nuisance_signed_score,

            "margin": margin,
            "reason": reason,
        })

    selection_df = pd.DataFrame(rows)

    selection_df = selection_df.sort_values(
        by=["remove", "nuisance_score", "margin"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    remove_df = selection_df[selection_df["remove"]].copy()

    if max_remove is not None:
        remove_df = remove_df.head(max_remove)

    remove_pc_list = remove_df["pc_number"].astype(int).tolist()
    remove_pc_names = remove_df["PC"].tolist()

    return {
        "remove_pc_list": remove_pc_list,
        "remove_pc_names": remove_pc_names,
        "selection_df": selection_df,
    }


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score


def _calc_dr_classification_metrics(y_true, y_pred, class_labels=None) -> dict:
    """
    计算DR等级分类的评价指标。
    主指标：QWK
    辅助指标：Accuracy / Macro-F1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if class_labels is None:
        class_labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))

    return {
        "qwk": cohen_kappa_score(
            y_true,
            y_pred,
            labels=class_labels,
            weights="quadratic",
        ),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=class_labels,
            average="macro",
            zero_division=0,
        ),
    }


def _remove_pcs_from_standardized_features(
    X_std: np.ndarray,
    pca,
    remove_pc_list: list[int] | int | None,
) -> np.ndarray:
    """
    在标准化后的特征空间中去除指定PC成分。

    注意：
        PC编号使用1始まり：
        remove_pc_list=[5, 6] 表示去除 PC5 和 PC6。
    """
    if remove_pc_list is None:
        return X_std.copy()

    if isinstance(remove_pc_list, int):
        remove_pc_list = [remove_pc_list]

    if len(remove_pc_list) == 0:
        return X_std.copy()

    indices = _normalize_pc_list(remove_pc_list, pca.n_components_)

    Z = pca.transform(X_std)
    V = pca.components_[indices]

    # 被去除PC对应的射影成分
    X_proj = Z[:, indices] @ V

    # 原始标准化特徴 - 非病変PC射影
    X_removed = X_std - X_proj

    return X_removed


def _make_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    生成交叉验证划分。

    如果 groups 不为 None，则优先使用 StratifiedGroupKFold。
    如果当前 sklearn 版本不支持，或者划分失败，则回退到 StratifiedKFold。
    """
    if groups is not None:
        try:
            from sklearn.model_selection import StratifiedGroupKFold

            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
            splits = list(splitter.split(X, y, groups))
            split_mode = "StratifiedGroupKFold"
            return splits, split_mode

        except Exception as e:
            print(f"StratifiedGroupKFold 无法使用，回退到 StratifiedKFold。原因: {e}")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(splitter.split(X, y))
    split_mode = "StratifiedKFold"

    return splits, split_mode


def summarize_pc_removal_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    """
    汇总每个fold的去除前后性能。
    """
    metrics = ["qwk", "accuracy", "macro_f1"]

    row = {}

    for metric in metrics:
        raw_col = f"raw_{metric}"
        clean_col = f"clean_{metric}"
        delta_col = f"delta_{metric}"

        row[f"{raw_col}_mean"] = fold_df[raw_col].mean()
        row[f"{raw_col}_std"] = fold_df[raw_col].std()

        row[f"{clean_col}_mean"] = fold_df[clean_col].mean()
        row[f"{clean_col}_std"] = fold_df[clean_col].std()

        row[f"{delta_col}_mean"] = fold_df[delta_col].mean()
        row[f"{delta_col}_std"] = fold_df[delta_col].std()

    row["n_removed"] = fold_df["n_removed"].iloc[0]
    row["removed_pcs"] = fold_df["removed_pcs"].iloc[0]
    row["split_mode"] = fold_df["split_mode"].iloc[0]

    if "model_name" in fold_df.columns:
        row["model_name"] = fold_df["model_name"].iloc[0]

    return pd.DataFrame([row])


def evaluate_pc_removal_effect(
    X: np.ndarray,
    meta_df: pd.DataFrame,
    y_col: str,
    remove_pc_list: list[int] | int | None,
    model_name: str | None = None,
    n_components: int = 50,
    n_splits: int = 5,
    group_col: str | None = None,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 100000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    比较去除指定PC前后的下流分类器性能。

    流程：
        1. 按fold划分 train / valid
        2. 只在train上fit StandardScaler
        3. 只在train上fit PCA
        4. raw: 用标准化后的原始特征训练分类器
        5. clean: 去除指定PC后的特征训练分类器
        6. 在同一个valid fold上比较 QWK / Accuracy / Macro-F1

    参数：
        X:
            原始特征向量矩阵。

        meta_df:
            metadata。

        y_col:
            DR等级列名。
            推荐先用 "eye_DR_Level"。
            如果你想做患者级严重度，则用 "patient_DR_Level"。

        remove_pc_list:
            需要去除的PC编号。
            例：[5, 6]

        group_col:
            如果想按患者分组避免同一患者进入train和valid两边，用 "patient_id"。
            如果坚持图像级随机划分，用 None。
    """
    X = np.asarray(X)
    meta_df = meta_df.reset_index(drop=True).copy()

    if len(X) != len(meta_df):
        raise ValueError(f"X 和 meta_df 行数不一致: X={len(X)}, meta_df={len(meta_df)}")

    if y_col not in meta_df.columns:
        raise KeyError(f"meta_df 中不存在 y_col: {y_col}")

    valid_mask = meta_df[y_col].notna()

    if group_col is not None:
        if group_col not in meta_df.columns:
            raise KeyError(f"meta_df 中不存在 group_col: {group_col}")
        valid_mask = valid_mask & meta_df[group_col].notna()

    X_use = X[valid_mask.to_numpy()]
    y_use = meta_df.loc[valid_mask, y_col].astype(int).to_numpy()

    if group_col is not None:
        groups = meta_df.loc[valid_mask, group_col].to_numpy()
    else:
        groups = None

    class_labels = np.sort(np.unique(y_use))

    splits, split_mode = _make_cv_splits(
        X=X_use,
        y=y_use,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
    )

    if isinstance(remove_pc_list, int):
        remove_pc_list = [remove_pc_list]
    elif remove_pc_list is None:
        remove_pc_list = []

    rows = []

    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        X_train = X_use[train_idx]
        X_valid = X_use[valid_idx]
        y_train = y_use[train_idx]
        y_valid = y_use[valid_idx]

        # 只在train上fit scaler，避免验证集泄露
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_valid_std = scaler.transform(X_valid)

        # 只在train上fit PCA，避免验证集泄露
        n_components_fold = min(
            n_components,
            X_train_std.shape[0],
            X_train_std.shape[1],
        )

        pca = PCA(n_components=n_components_fold)
        pca.fit(X_train_std)

        # 当前fold中可用的去除PC
        remove_pc_list_fold = [
            pc for pc in remove_pc_list
            if pc <= n_components_fold
        ]

        # 原始特征空间分类器
        raw_clf = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )
        raw_clf.fit(X_train_std, y_train)
        y_pred_raw = raw_clf.predict(X_valid_std)

        # 去除非病変PC后的特征空间
        X_train_clean = _remove_pcs_from_standardized_features(
            X_std=X_train_std,
            pca=pca,
            remove_pc_list=remove_pc_list_fold,
        )
        X_valid_clean = _remove_pcs_from_standardized_features(
            X_std=X_valid_std,
            pca=pca,
            remove_pc_list=remove_pc_list_fold,
        )

        clean_clf = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )
        clean_clf.fit(X_train_clean, y_train)
        y_pred_clean = clean_clf.predict(X_valid_clean)

        raw_metrics = _calc_dr_classification_metrics(
            y_true=y_valid,
            y_pred=y_pred_raw,
            class_labels=class_labels,
        )
        clean_metrics = _calc_dr_classification_metrics(
            y_true=y_valid,
            y_pred=y_pred_clean,
            class_labels=class_labels,
        )

        row = {
            "model_name": model_name,
            "fold": fold_id,
            "split_mode": split_mode,
            "n_train": len(train_idx),
            "n_valid": len(valid_idx),
            "n_components": n_components_fold,
            "removed_pcs": ",".join([f"PC{pc}" for pc in remove_pc_list_fold]),
            "n_removed": len(remove_pc_list_fold),
        }

        for metric in ["qwk", "accuracy", "macro_f1"]:
            row[f"raw_{metric}"] = raw_metrics[metric]
            row[f"clean_{metric}"] = clean_metrics[metric]
            row[f"delta_{metric}"] = clean_metrics[metric] - raw_metrics[metric]

        rows.append(row)

    fold_df = pd.DataFrame(rows)
    summary_df = summarize_pc_removal_results(fold_df)

    return fold_df, summary_df


def plot_pc_removal_comparison(
    fold_df: pd.DataFrame,
    metric: str = "qwk",
    title: str | None = None,
    figsize: tuple[int, int] = (6, 4),
):
    """
    可视化去除PC前后的性能变化。
    metric:
        "qwk", "accuracy", "macro_f1"
    """
    raw_col = f"raw_{metric}"
    clean_col = f"clean_{metric}"

    plot_df = fold_df[["fold", raw_col, clean_col]].melt(
        id_vars="fold",
        value_vars=[raw_col, clean_col],
        var_name="condition",
        value_name=metric,
    )

    plot_df["condition"] = plot_df["condition"].map({
        raw_col: "raw",
        clean_col: "pc_removed",
    })

    plt.figure(figsize=figsize)

    sns.barplot(
        data=plot_df,
        x="condition",
        y=metric,
        errorbar="sd",
    )

    sns.stripplot(
        data=plot_df,
        x="condition",
        y=metric,
        color="black",
        size=5,
        alpha=0.7,
    )

    if title is None:
        title = f"Before vs After PC removal ({metric})"

    plt.title(title)
    plt.xlabel("")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()



def summarize_auto_pc_removal_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    """
    汇总 fold 内自动选择 PC 后的性能结果。
    注意：
        这里每个 fold 去除的 PC 可能不同，因此不再假设 removed_pcs 固定。
    """
    if len(fold_df) == 0:
        return pd.DataFrame()

    metrics = ["qwk", "accuracy", "macro_f1"]
    row = {}

    for metric in metrics:
        raw_col = f"raw_{metric}"
        clean_col = f"clean_{metric}"
        delta_col = f"delta_{metric}"

        row[f"{raw_col}_mean"] = fold_df[raw_col].mean()
        row[f"{raw_col}_std"] = fold_df[raw_col].std()

        row[f"{clean_col}_mean"] = fold_df[clean_col].mean()
        row[f"{clean_col}_std"] = fold_df[clean_col].std()

        row[f"{delta_col}_mean"] = fold_df[delta_col].mean()
        row[f"{delta_col}_std"] = fold_df[delta_col].std()

    row["n_removed_mean"] = fold_df["n_removed"].mean()
    row["n_removed_std"] = fold_df["n_removed"].std()
    row["n_removed_min"] = fold_df["n_removed"].min()
    row["n_removed_max"] = fold_df["n_removed"].max()

    row["removed_pcs_by_fold"] = " | ".join(
        [
            f"fold{int(r.fold)}:{r.removed_pcs}"
            for r in fold_df.itertuples()
        ]
    )

    row["split_mode"] = fold_df["split_mode"].iloc[0]

    if "model_name" in fold_df.columns:
        row["model_name"] = fold_df["model_name"].iloc[0]

    return pd.DataFrame([row])


def _filter_auc_cols_for_fold(meta_train_df: pd.DataFrame, auc_cols: list[str]) -> list[str]:
    """
    只保留当前 train fold 中可以做二分类 AUC 的列。
    """
    usable_cols = []

    for col in auc_cols:
        if col not in meta_train_df.columns:
            continue

        n_unique = meta_train_df[col].dropna().nunique()

        if n_unique == 2:
            usable_cols.append(col)

    return usable_cols


def _filter_spearman_cols_for_fold(meta_train_df: pd.DataFrame, spearman_cols: list[str]) -> list[str]:
    """
    只保留当前 train fold 中可以做 Spearman 的列。
    """
    usable_cols = []

    for col in spearman_cols:
        if col not in meta_train_df.columns:
            continue

        n_unique = meta_train_df[col].dropna().nunique()

        if n_unique >= 2:
            usable_cols.append(col)

    return usable_cols


def evaluate_pc_removal_effect_auto_fold(
    X: np.ndarray,
    meta_df: pd.DataFrame,
    y_col: str = "eye_DR_Level",
    model_name: str | None = None,
    n_components: int = 50,
    n_splits: int = 5,

    auc_cols: list[str] | None = None,
    spearman_cols: list[str] | None = None,
    disease_cols: list[str] | None = None,
    nuisance_cols: list[str] | None = None,
    positive_label_map: dict | None = None,

    nuisance_threshold: float = 0.30,
    disease_threshold: float = 0.20,
    margin_threshold: float = 0.10,
    max_remove: int | None = 3,

    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 200000,
    random_state: int = 42,

    add_view_no: bool = True,
    image_path_col: str = "image_path",
    view_col: str = "view_no",

    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fold 内自动选择并去除 PC，然后评价分类性能。

    这是比固定 remove_pc_list 更严格的做法。

    每个 fold 的流程：
        1. 图像级 StratifiedKFold 划分 train / valid
        2. 只在 train fold 上 fit StandardScaler
        3. 只在 train fold 上 fit PCA
        4. 只用 train fold 的 PC score 与 metadata 计算 PC-factor score
        5. 只根据 train fold 自动选择要去除的 PC
        6. 用同一个 train-fitted scaler / PCA 去除 train 和 valid 的对应 PC
        7. raw 特征训练分类器
        8. PC removed 特征训练分类器
        9. 在 valid fold 上比较 QWK / Accuracy / Macro-F1

    重要：
        - 不使用 patient_id
        - 不做 patient-level split
        - valid fold 的标签不会参与 PC 选择
    """
    X = np.asarray(X)
    meta_df = meta_df.reset_index(drop=True).copy()

    if len(X) != len(meta_df):
        raise ValueError(f"X 和 meta_df 行数不一致: X={len(X)}, meta_df={len(meta_df)}")

    if y_col not in meta_df.columns:
        raise KeyError(f"meta_df 中不存在 y_col: {y_col}")

    if auc_cols is None:
        auc_cols = [
            "Overall quality",
            "eye",
            "view_no",
        ]

    if spearman_cols is None:
        spearman_cols = [
            y_col,
            "Clarity",
            "Field definition",
            "Artifact",
        ]

    # 保证疾病标签一定进入 Spearman 评价，用来防止误删疾病轴
    if y_col not in spearman_cols:
        spearman_cols = [y_col] + spearman_cols

    if disease_cols is None:
        disease_cols = [y_col]

    if nuisance_cols is None:
        nuisance_cols = [
            "Overall quality",
            "eye",
            "view_no",
            "Clarity",
            "Field definition",
            "Artifact",
        ]

    if positive_label_map is None:
        positive_label_map = {
            "Overall quality": 1,
            "eye": "r",
            "view_no": 2,
        }

    # 如果需要 view_no，但是 meta_df 里没有，则从 image_path 推断
    need_view_no = (
        view_col in auc_cols
        or view_col in nuisance_cols
        or view_col in disease_cols
        or view_col in spearman_cols
    )

    if add_view_no and need_view_no and view_col not in meta_df.columns:
        if image_path_col in meta_df.columns:
            meta_df = add_view_no_column(
                meta_df=meta_df,
                image_path_col=image_path_col,
                view_col=view_col,
                overwrite=False,
                strict=False,
            )
        else:
            if verbose:
                print(f"警告：meta_df 中没有 {image_path_col}，无法自动生成 {view_col}。")

    valid_mask = meta_df[y_col].notna()

    X_use = X[valid_mask.to_numpy()]
    meta_use = meta_df.loc[valid_mask].reset_index(drop=True).copy()
    y_use = meta_use[y_col].astype(int).to_numpy()

    class_labels = np.sort(np.unique(y_use))

    splits, split_mode = _make_cv_splits(
        X=X_use,
        y=y_use,
        groups=None,
        n_splits=n_splits,
        random_state=random_state,
    )

    fold_rows = []
    selection_rows = []
    score_rows = []

    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        if verbose:
            print(f"===== {model_name} | Fold {fold_id} =====")

        X_train = X_use[train_idx]
        X_valid = X_use[valid_idx]
        y_train = y_use[train_idx]
        y_valid = y_use[valid_idx]

        meta_train = meta_use.iloc[train_idx].reset_index(drop=True).copy()

        # 只在 train fold 上 fit scaler
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_valid_std = scaler.transform(X_valid)

        # 只在 train fold 上 fit PCA
        n_components_fold = min(
            n_components,
            X_train_std.shape[0],
            X_train_std.shape[1],
        )

        pca = PCA(n_components=n_components_fold)
        pca.fit(X_train_std)

        # 只用 train fold 的 PC score 做 PC-factor 关系分析
        Z_train = pca.transform(X_train_std)
        pca_df_train = make_pca_df(Z_train)

        fold_auc_cols = _filter_auc_cols_for_fold(meta_train, auc_cols)
        fold_spearman_cols = _filter_spearman_cols_for_fold(meta_train, spearman_cols)

        if len(fold_auc_cols) == 0 and len(fold_spearman_cols) == 0:
            raise ValueError(
                f"Fold {fold_id}: 没有可用于 PC-factor 计算的因子列。"
            )

        score_df_train = compute_pc_factor_score_table(
            pca_df=pca_df_train,
            meta_df=meta_train,
            auc_cols=fold_auc_cols,
            spearman_cols=fold_spearman_cols,
            n_pc=n_components_fold,
            positive_label_map=positive_label_map,
        )

        score_df_train["model_name"] = model_name
        score_df_train["fold"] = fold_id
        score_df_train["split_mode"] = split_mode
        score_rows.append(score_df_train)

        existing_factors = set(score_df_train["factor"].unique())

        disease_cols_fold = [
            c for c in disease_cols
            if c in existing_factors
        ]

        nuisance_cols_fold = [
            c for c in nuisance_cols
            if c in existing_factors
        ]

        if len(disease_cols_fold) == 0:
            raise ValueError(
                f"Fold {fold_id}: disease_cols 中没有任何列进入 score_df。"
                f"当前 disease_cols={disease_cols}, score_df factors={sorted(existing_factors)}"
            )

        if len(nuisance_cols_fold) == 0:
            raise ValueError(
                f"Fold {fold_id}: nuisance_cols 中没有任何列进入 score_df。"
                f"当前 nuisance_cols={nuisance_cols}, score_df factors={sorted(existing_factors)}"
            )

        # 当前 fold 内自动选择要去除的 PC
        selection_result = select_pcs_to_remove(
            score_df=score_df_train,
            disease_cols=disease_cols_fold,
            nuisance_cols=nuisance_cols_fold,
            nuisance_threshold=nuisance_threshold,
            disease_threshold=disease_threshold,
            margin_threshold=margin_threshold,
            max_remove=max_remove,
        )

        remove_pc_list_fold = selection_result["remove_pc_list"]
        remove_pc_names_fold = selection_result["remove_pc_names"]

        selection_df_fold = selection_result["selection_df"].copy()
        selection_df_fold["model_name"] = model_name
        selection_df_fold["fold"] = fold_id
        selection_df_fold["split_mode"] = split_mode
        selection_df_fold["nuisance_threshold"] = nuisance_threshold
        selection_df_fold["disease_threshold"] = disease_threshold
        selection_df_fold["margin_threshold"] = margin_threshold
        selection_rows.append(selection_df_fold)

        if verbose:
            print(f"  selected PCs: {remove_pc_names_fold}")

        # raw 分类器：标准化后的原始特征
        raw_clf = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )
        raw_clf.fit(X_train_std, y_train)
        y_pred_raw = raw_clf.predict(X_valid_std)

        # clean 分类器：去除当前 fold 自动选出的 PC
        X_train_clean = _remove_pcs_from_standardized_features(
            X_std=X_train_std,
            pca=pca,
            remove_pc_list=remove_pc_list_fold,
        )

        X_valid_clean = _remove_pcs_from_standardized_features(
            X_std=X_valid_std,
            pca=pca,
            remove_pc_list=remove_pc_list_fold,
        )

        clean_clf = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )
        clean_clf.fit(X_train_clean, y_train)
        y_pred_clean = clean_clf.predict(X_valid_clean)

        raw_metrics = _calc_dr_classification_metrics(
            y_true=y_valid,
            y_pred=y_pred_raw,
            class_labels=class_labels,
        )

        clean_metrics = _calc_dr_classification_metrics(
            y_true=y_valid,
            y_pred=y_pred_clean,
            class_labels=class_labels,
        )

        if len(remove_pc_names_fold) == 0:
            removed_pcs_text = "None"
        else:
            removed_pcs_text = ",".join(remove_pc_names_fold)

        row = {
            "model_name": model_name,
            "fold": fold_id,
            "split_mode": split_mode,
            "n_train": len(train_idx),
            "n_valid": len(valid_idx),
            "n_components": n_components_fold,
            "removed_pcs": removed_pcs_text,
            "n_removed": len(remove_pc_list_fold),
        }

        for metric in ["qwk", "accuracy", "macro_f1"]:
            row[f"raw_{metric}"] = raw_metrics[metric]
            row[f"clean_{metric}"] = clean_metrics[metric]
            row[f"delta_{metric}"] = clean_metrics[metric] - raw_metrics[metric]

        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    summary_df = summarize_auto_pc_removal_results(fold_df)

    if len(selection_rows) > 0:
        selection_df = pd.concat(selection_rows, axis=0, ignore_index=True)
    else:
        selection_df = pd.DataFrame()

    if len(score_rows) > 0:
        score_df_cv = pd.concat(score_rows, axis=0, ignore_index=True)
    else:
        score_df_cv = pd.DataFrame()

    return fold_df, summary_df, selection_df, score_df_cv   
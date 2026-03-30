import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


Array4D = np.ndarray  # [Nx, Ny, Nz, C]
Array2D = np.ndarray  # wells: [N, 4+] -> x,y,z,label,(optional well_id)

DEFAULT_FILE_PATH_MARKS = "W:/3d/kovikta/Work/Well_new_pogl_filled.csv"
DEFAULT_FILE_PATH_NOT_MARKS = "W:/3d/kovikta/Work/New_kov_pravki_more.txt"
FEATURES = [
    "Cube_X",
    "Cube_Y",
    "Cube_Z",
    "layer",
    "Rho",
    "S",
    "thgr",
    "Sgr",
    "krgr",
    "fml",
    "kp",
]


@dataclass
class GridSpec:
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


def load_kovikta_dataframes(
    file_path_marks: str = DEFAULT_FILE_PATH_MARKS,
    file_path_not_marks: str = DEFAULT_FILE_PATH_NOT_MARKS,
):
    """Загрузка табличных данных в стиле:

    df_labeled = pd.read_csv(file_path_marks)
    df_unlabeled = pd.read_csv(file_path_not_marks, sep="\\t")
    df_unlabeled.rename({"X":"Cube_X","Y":"Cube_Y","Z":"Cube_Z"})
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Для загрузки CSV/TXT таблиц требуется pandas.") from exc

    df_labeled = pd.read_csv(file_path_marks)
    df_unlabeled = pd.read_csv(file_path_not_marks, sep="\t")
    df_unlabeled = df_unlabeled.rename(columns={"X": "Cube_X", "Y": "Cube_Y", "Z": "Cube_Z"})

    missing_unlabeled = [c for c in FEATURES if c not in df_unlabeled.columns]
    if missing_unlabeled:
        raise ValueError(f"В unlabeled-таблице отсутствуют колонки: {missing_unlabeled}")

    return df_labeled, df_unlabeled


def wells_from_labeled_dataframe(df_labeled, label_col: str) -> Array2D:
    """Конвертирует DataFrame -> wells ndarray [x,y,z,label]."""
    required = ["Cube_X", "Cube_Y", "Cube_Z", label_col]
    missing = [c for c in required if c not in df_labeled.columns]
    if missing:
        raise ValueError(f"В labeled-таблице отсутствуют колонки: {missing}")

    wells = (
        df_labeled[["Cube_X", "Cube_Y", "Cube_Z", label_col]]
        .dropna(subset=["Cube_X", "Cube_Y", "Cube_Z", label_col])
        .to_numpy(np.float32)
    )
    return wells


def load_cube_txt(path: str, nx: int, ny: int, nz: int, c: int, dtype=np.float32) -> Array4D:
    """Загружает геокуб из txt и приводит к форме [Nx, Ny, Nz, C]."""
    raw = np.loadtxt(path, dtype=dtype)
    expected = nx * ny * nz * c
    if raw.size != expected:
        raise ValueError(f"Размер cube txt не совпадает: {raw.size=} != {expected=}")
    cube = raw.reshape(nx, ny, nz, c)
    return cube


def load_wells_txt(path: str, dtype=np.float32) -> Array2D:
    """Загружает скважины [x, y, z, label] (и опционально well_id в 5-й колонке)."""
    wells = np.loadtxt(path, dtype=dtype)
    if wells.ndim == 1:
        wells = wells[None, :]
    if wells.shape[1] < 4:
        raise ValueError("Ожидается минимум 4 колонки: x, y, z, label")
    return wells


def world_to_index(coords_xyz: np.ndarray, grid: GridSpec) -> np.ndarray:
    """(x,y,z) -> (i,j,k), округление к ближайшему индексу."""
    origin = np.asarray(grid.origin, dtype=np.float32)
    spacing = np.asarray(grid.spacing, dtype=np.float32)
    idx = np.rint((coords_xyz - origin) / spacing).astype(np.int64)
    return idx


def validate_wells_in_cube(indices_ijk: np.ndarray, cube_shape: Sequence[int]) -> np.ndarray:
    """Возвращает булеву маску точек, которые попадают в границы куба."""
    nx, ny, nz = cube_shape[:3]
    i_ok = (0 <= indices_ijk[:, 0]) & (indices_ijk[:, 0] < nx)
    j_ok = (0 <= indices_ijk[:, 1]) & (indices_ijk[:, 1] < ny)
    k_ok = (0 <= indices_ijk[:, 2]) & (indices_ijk[:, 2] < nz)
    return i_ok & j_ok & k_ok


def normalize_cube(cube: Array4D, mode: str = "zscore") -> Tuple[Array4D, NormalizationStats]:
    """
    Нормализация по каналам:
    - zscore: (k-mean)/std
    - minmax: (k-min)/(max-min)
    NaN игнорируются при вычислении статистик.
    """
    c = cube.shape[-1]
    out = cube.astype(np.float32).copy()
    mean = np.zeros(c, dtype=np.float32)
    std = np.ones(c, dtype=np.float32)

    for ch in range(c):
        channel = out[..., ch]
        valid = ~np.isnan(channel)
        if valid.sum() == 0:
            continue

        if mode == "zscore":
            m = channel[valid].mean()
            s = channel[valid].std()
            s = max(s, 1e-8)
            out[..., ch] = (channel - m) / s
            mean[ch], std[ch] = m, s
        elif mode == "minmax":
            mn = channel[valid].min()
            mx = channel[valid].max()
            rng = max(mx - mn, 1e-8)
            out[..., ch] = (channel - mn) / rng
            mean[ch], std[ch] = mn, rng
        else:
            raise ValueError("mode должен быть 'zscore' или 'minmax'")

    return out, NormalizationStats(mean=mean, std=std)


def fill_nan(cube: Array4D, strategy: str = "channel_mean") -> Array4D:
    out = cube.copy()
    if strategy == "channel_mean":
        for ch in range(out.shape[-1]):
            x = out[..., ch]
            m = np.nanmean(x)
            x[np.isnan(x)] = m
            out[..., ch] = x
        return out
    if strategy == "zero":
        out[np.isnan(out)] = 0.0
        return out
    raise ValueError("strategy должен быть 'channel_mean' или 'zero'")


def split_by_wells(
    wells: Array2D,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, Array2D]:
    """Разделение по группам скважин (не случайно по отдельным точкам).

    Если есть 5-я колонка, используется как well_id.
    Иначе group_id формируется по уникальным (x, y).
    """
    rng = np.random.default_rng(random_seed)

    if wells.shape[1] >= 5:
        group_id = wells[:, 4].astype(np.int64)
    else:
        xy = wells[:, :2]
        _, group_id = np.unique(xy, axis=0, return_inverse=True)

    unique_ids = np.unique(group_id)
    rng.shuffle(unique_ids)

    n_total = len(unique_ids)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio)) if n_total > 2 else 0

    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train : n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val :])

    train = wells[np.isin(group_id, list(train_ids))]
    val = wells[np.isin(group_id, list(val_ids))] if n_val > 0 else np.empty((0, wells.shape[1]))
    test = wells[np.isin(group_id, list(test_ids))]

    return {"train_wells": train, "val_wells": val, "test_wells": test}


def class_weights(labels: np.ndarray) -> torch.Tensor:
    classes, counts = np.unique(labels.astype(np.int64), return_counts=True)
    total = counts.sum()
    weights = np.zeros(classes.max() + 1, dtype=np.float32)
    for cls, cnt in zip(classes, counts):
        weights[cls] = total / (len(classes) * cnt)
    return torch.tensor(weights, dtype=torch.float32)


def extract_patch(cube: Array4D, center_ijk: Tuple[int, int, int], patch_size: Tuple[int, int, int], pad_mode: str = "reflect") -> Array4D:
    px, py, pz = patch_size
    hx, hy, hz = px // 2, py // 2, pz // 2
    i, j, k = center_ijk

    padded = np.pad(
        cube,
        pad_width=((hx, hx), (hy, hy), (hz, hz), (0, 0)),
        mode=pad_mode,
    )
    ip, jp, kp = i + hx, j + hy, k + hz
    patch = padded[ip - hx : ip + hx + 1, jp - hy : jp + hy + 1, kp - hz : kp + hz + 1, :]
    return patch


class GeoDataset(Dataset):
    """Каждая точка скважины -> patch + метка."""

    def __init__(
        self,
        cube: Array4D,
        wells: Array2D,
        patch_size: Tuple[int, int, int],
        grid: GridSpec,
        pad_mode: str = "reflect",
        skip_outside: bool = True,
    ):
        self.cube = cube
        self.patch_size = patch_size
        self.pad_mode = pad_mode

        coords_xyz = wells[:, :3]
        labels = wells[:, 3].astype(np.int64)

        ijk = world_to_index(coords_xyz, grid)
        in_bounds = validate_wells_in_cube(ijk, cube.shape)

        if skip_outside:
            ijk = ijk[in_bounds]
            labels = labels[in_bounds]
        else:
            ijk = np.clip(ijk, [0, 0, 0], np.array(cube.shape[:3]) - 1)

        self.ijk = ijk
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        i, j, k = self.ijk[idx]
        patch = extract_patch(self.cube, (int(i), int(j), int(k)), self.patch_size, self.pad_mode)
        patch = np.transpose(patch, (3, 0, 1, 2))  # [C, Px, Py, Pz]
        x = torch.tensor(patch, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class Simple3DCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.flatten(1)
        return self.classifier(h)


def predict_probability_cube(model: nn.Module, cube: Array4D, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    """Предсказание вероятности положительного класса для всех узлов куба.

    Для многокласса возвращается max probability по классам.
    """
    model.eval()
    nx, ny, nz, c = cube.shape
    flat = cube.reshape(-1, c)
    x = torch.tensor(flat, dtype=torch.float32, device=device).view(-1, c, 1, 1, 1)

    probs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            logits = model(x[start : start + batch_size])
            p = torch.softmax(logits, dim=1)
            if p.shape[1] == 2:
                p = p[:, 1]
            else:
                p = p.max(dim=1).values
            probs.append(p.cpu().numpy())

    return np.concatenate(probs).reshape(nx, ny, nz)


def main():
    parser = argparse.ArgumentParser(description="3D CNN классификация геокуба")
    parser.add_argument("--cube_txt")
    parser.add_argument("--wells_txt")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--nz", type=int)
    parser.add_argument("--c", type=int)
    parser.add_argument("--patch", type=int, nargs=3, default=[5, 5, 5])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--norm", choices=["zscore", "minmax"], default="zscore")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--marks_csv", default=DEFAULT_FILE_PATH_MARKS)
    parser.add_argument("--unlabeled_txt", default=DEFAULT_FILE_PATH_NOT_MARKS)
    parser.add_argument("--label_col", default="label")
    parser.add_argument(
        "--prepare_kovikta_only",
        action="store_true",
        help="Только загрузить/провалидировать таблицы marks + not_marks и сохранить wells.txt",
    )
    raw_argv = [x for x in sys.argv[1:] if x not in {"\\n", "`n"}]
    args, unknown = parser.parse_known_args(raw_argv)
    if unknown:
        raise ValueError(
            "Неизвестные аргументы: "
            f"{unknown}. Если вы запускаете в PowerShell, не используйте '\\n' для переноса строки. "
            "Используйте одну строку или перенос через обратную кавычку (`)."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare_kovikta_only:
        df_labeled, df_unlabeled = load_kovikta_dataframes(args.marks_csv, args.unlabeled_txt)
        wells = wells_from_labeled_dataframe(df_labeled, args.label_col)
        wells_path = out_dir / "wells_from_marks.txt"
        np.savetxt(wells_path, wells, fmt="%.6f")
        unlabeled_path = out_dir / "unlabeled_features_preview.csv"
        df_unlabeled[FEATURES].to_csv(unlabeled_path, index=False)
        print(f"Saved: {wells_path}")
        print(f"Saved: {unlabeled_path}")
        return

    required_train_args = [args.cube_txt, args.wells_txt, args.nx, args.ny, args.nz, args.c]
    if any(v is None for v in required_train_args):
        raise ValueError(
            "Для обучения укажите --cube_txt --wells_txt --nx --ny --nz --c "
            "или используйте --prepare_kovikta_only."
        )

    cube = load_cube_txt(args.cube_txt, args.nx, args.ny, args.nz, args.c)
    wells = load_wells_txt(args.wells_txt)

    cube, stats = normalize_cube(cube, mode=args.norm)
    cube = fill_nan(cube, strategy="channel_mean")

    splits = split_by_wells(wells)
    grid = GridSpec()

    train_ds = GeoDataset(cube, splits["train_wells"], tuple(args.patch), grid)
    val_ds = GeoDataset(cube, splits["val_wells"], tuple(args.patch), grid)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if len(val_ds) > 0 else None

    num_classes = int(np.max(wells[:, 3])) + 1
    model = Simple3DCNN(in_channels=args.c, num_classes=num_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    weights = class_weights(train_ds.labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        msg = f"epoch={epoch+1}/{args.epochs} train_loss={total_loss / max(1, len(train_loader)):.4f}"

        if val_loader is not None:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    logits = model(batch_x.to(device))
                    pred = logits.argmax(dim=1).cpu()
                    correct += int((pred == batch_y).sum().item())
                    total += int(batch_y.numel())
            acc = correct / max(1, total)
            msg += f" val_acc={acc:.4f}"
        print(msg)

    # 1) модель
    model_path = out_dir / "model.pth"
    torch.save(model.state_dict(), model_path)

    # 2) нормализация
    norm_path = out_dir / "norm_stats.npz"
    np.savez(norm_path, mean=stats.mean, std=stats.std)

    # 3) probability cube [Nx,Ny,Nz] -> txt
    prob_cube = predict_probability_cube(model, cube, device=device)
    prob_path = out_dir / "probability_cube.txt"
    np.savetxt(prob_path, prob_cube.reshape(-1), fmt="%.6f")

    print(f"Saved: {model_path}")
    print(f"Saved: {norm_path}")
    print(f"Saved: {prob_path}")


if __name__ == "__main__":
    main()

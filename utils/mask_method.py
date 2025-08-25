import torch


def _make_generator(device, seed):
    g = torch.Generator()
    try:
        g = torch.Generator(device=device)
    except TypeError:
        pass
    g.manual_seed(seed)
    return g

@torch.no_grad()
def generate_mar_mask(
    X: torch.Tensor,           # [B,T,D]
    f_dim: int,                # 最后 f_dim 是目标特征，全部做 MAR
    obs_rate: float = 0.5,     # 在非目标列里选取的预测器比例 (0,1]；=1 表示全部非目标列都保留观测并用作预测器
    missing_rate=0.2,          # 目标列的缺失率：标量或长度=f_dim 的 1D 张量/列表
    strict: bool = False,       # True: 预测器列永远观测（严格 MAR）；False: 放宽（不建议用于“严格”定义）
    max_iter: int = 30,
    seed: int = 42,
    eps: float = 1e-8,
):
    assert X.ndim == 3, "X 必须是 [batch, seq, feature]"
    B, T, D = X.shape
    assert 1 <= f_dim <= D, "f_dim 范围应为 [1, D]"
    ctx = D - f_dim                      # 非目标列数量
    if strict and ctx == 0:
        raise ValueError("strict MAR 需要至少 1 个非目标列作为预测器（D - f_dim > 0）。")

    if isinstance(missing_rate, (int, float)):
        assert 0 < missing_rate < 1, "missing_rate 应在 (0,1)"
    else:
        # 各目标列各自的缺失率
        mr = torch.as_tensor(missing_rate, dtype=torch.float32)
        if mr.numel() != f_dim or not torch.all((mr > 0) & (mr < 1)):
            raise ValueError("missing_rate 需为标量或长度=f_dim 的 (0,1) 向量。")

    device = X.device
    N = B * T
    X2d = X.reshape(N, D).float()
    if not torch.isfinite(X2d).all():
        raise ValueError("X 含 NaN/Inf，请先清洗。")

    # 标准化，避免数值不稳
    mu = X2d.mean(dim=0)
    sd = X2d.std(dim=0)
    sd = torch.where(sd < eps, torch.ones_like(sd), sd)
    Xz = (X2d - mu) / sd

    gen = _make_generator(device, seed)

    # 索引
    target_idx = torch.arange(D - f_dim, D, device=device)       # 目标：最后 f_dim 列
    ctx_idx = torch.arange(0, ctx, device=device) if ctx > 0 else torch.tensor([], device=device)

    # 把 missing_rate 变成长度 f_dim 的向量
    def _mr_vec():
        if isinstance(missing_rate, (int, float)):
            return torch.full((f_dim,), float(missing_rate), device=device)
        return torch.as_tensor(missing_rate, device=device, dtype=X2d.dtype)

    if strict:
        # 从非目标列中选取预测器（并永久观测）
        assert 0 < obs_rate <= 1, "strict 模式下 obs_rate 应在 (0,1]"
        n_pred_pool = ctx_idx.numel()
        n_pred = max(1, int(round(obs_rate * n_pred_pool)))
        if n_pred > n_pred_pool:
            n_pred = n_pred_pool
        # 固定选择或随机选择都可以；用随机可复现
        if n_pred == n_pred_pool:
            pred_idx = ctx_idx
        else:
            perm = ctx_idx[torch.randperm(n_pred_pool, generator=gen, device=device)]
            pred_idx = perm[:n_pred]

        # 线性打分：仅用预测器 → 目标
        W = torch.randn(n_pred, f_dim, generator=gen, device=device) / (n_pred ** 0.5)
        S = Xz[:, pred_idx] @ W                                     # [N, f_dim]

        mr = _mr_vec()                                              # [f_dim]
        lo = torch.full((f_dim,), -20.0, device=device)
        hi = torch.full((f_dim,),  20.0, device=device)
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            p = torch.sigmoid(S + mid)
            m = p.mean(dim=0)
            hi = torch.where(m > mr, mid, hi)
            lo = torch.where(m <= mr, mid, lo)
        alpha = (lo + hi) / 2.0
        miss_prob_t = torch.sigmoid(S + alpha)                      # [N, f_dim]

        # 采样掩码（注意：对目标列）
        u = torch.rand(miss_prob_t.shape, device=device, dtype=miss_prob_t.dtype, generator=gen)
        mask_t = (u > miss_prob_t).float()                          # [N, f_dim]

        # 汇总成完整 mask：预测器列永远观测=1；目标列按 mask_t；其余非目标但未选为预测器的列也强制观测（严格 MAR）
        mask2d = torch.ones(N, D, device=device)
        mask2d[:, target_idx] = mask_t
        mask2d[:, ctx_idx] = 1.0                                    # 非目标列总体观测
        # 可选：如果你只想让“被选为预测器”的列保证观测，而其它非目标列也可以被随意保留观测，上一行保持即可。

        info = {
            "mode": "strict_tail_targets",
            "predictor_idx": pred_idx.tolist(),
            "target_idx": target_idx.tolist(),
        }

    else:
        # 非 strict：每个目标列依赖若干非目标列（若 ctx=0，则无法定义 MAR，直接报错以避免混淆）
        if ctx == 0:
            raise ValueError("non-strict 也需要至少 1 个非目标列来定义对‘已观测变量’的依赖；否则会变成 MNAR。")
        k = max(1, int(round(obs_rate * ctx)))   # 每个目标列的预测器个数
        W = torch.zeros(ctx, f_dim, device=device)
        for j in range(f_dim):
            perm = ctx_idx[torch.randperm(ctx, generator=gen, device=device)]
            sel = perm[:k]
            W[(sel - 0), j] = torch.randn(sel.numel(), generator=gen, device=device) / (k ** 0.5)

        S = Xz[:, ctx_idx] @ W                                       # [N, f_dim]
        mr = _mr_vec()
        lo = torch.full((f_dim,), -20.0, device=device)
        hi = torch.full((f_dim,),  20.0, device=device)
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            p = torch.sigmoid(S + mid)
            m = p.mean(dim=0)
            hi = torch.where(m > mr, mid, hi)
            lo = torch.where(m <= mr, mid, lo)
        alpha = (lo + hi) / 2.0
        miss_prob_t = torch.sigmoid(S + alpha)

        u = torch.rand(miss_prob_t.shape, device=device, dtype=miss_prob_t.dtype, generator=gen)
        mask_t = (u > miss_prob_t).float()

        mask2d = torch.ones(N, D, device=device)
        mask2d[:, target_idx] = mask_t
        # 非 strict 下，非目标列默认全观测（可视为“始终可用的外生变量”）
        mask2d[:, ctx_idx] = 1.0

        info = {
            "mode": "non_strict_tail_targets",
            "predictors_per_target": k,
            "target_idx": target_idx.tolist(),
        }

    # 还原形状并应用
    mask = mask2d.view(B, T, D)
    X_miss = X.clone()
    X_miss[mask == 0] = float("nan")

    # 统计
    overall = float((mask == 0).float().mean().cpu())
    target_rate = float((mask[:, :, target_idx] == 0).float().mean().cpu())
    info.update({
        "achieved_missing_rate_overall": overall,
        "achieved_missing_rate_targets": target_rate,
    })
    return X_miss


@torch.no_grad()
def generate_mcar_mask(
    X: torch.Tensor,                  # [B, T, D] 完整数据（float，不能含 NaN/Inf）
    missing_rate=0.2,                 # 缺失率：标量 (0,1) 或 1D 向量（见下）
    tail_targets_only: bool = False,  # True: 仅对最后 f_dim 列造缺
    f_dim: int | None = None,         # 当 tail_targets_only=True 时必须提供
    pattern: str = "cell",            # "cell" | "row"
    seed: int = 42,
):
    """
    返回:
      X_miss: NaN 表示缺失，shape=[B,T,D]
      mask:   观测指示 (1=观测, 0=缺失)，shape=[B,T,D]
      info:   元信息（实际缺失率/被造缺列/模式等）

    missing_rate 的用法：
      - 若对“全部特征”造缺：可为标量，或长度 D 的 1D 张量/列表（逐列缺失率）
      - 若仅对“最后 f_dim 列”造缺：可为标量，或长度 f_dim 的 1D 张量/列表
    """
    assert X.ndim == 3, "X 必须是 [batch, seq, feature]"
    device = X.device
    B, T, D = X.shape
    N = B * T

    X2d = X.reshape(N, D).float()
    if not torch.isfinite(X2d).all():
        raise ValueError("X 含 NaN/Inf，请先清洗。")

    # 选择要造缺的列索引
    if tail_targets_only:
        assert f_dim is not None and 1 <= f_dim <= D, "请提供有效的 f_dim"
        sel_idx = torch.arange(D - f_dim, D, device=device)   # 最后 f_dim 列
    else:
        sel_idx = torch.arange(0, D, device=device)           # 全部列
        f_dim = sel_idx.numel()

    # 统一 missing_rate 为长度=len(sel_idx) 的张量
    if isinstance(missing_rate, (list, tuple)):
        p_vec = torch.tensor(missing_rate, dtype=torch.float32, device=device)
    elif isinstance(missing_rate, torch.Tensor):
        p_vec = missing_rate.to(device=device, dtype=torch.float32)
    else:  # 标量
        p_vec = torch.full((sel_idx.numel(),), float(missing_rate),
                           device=device, dtype=torch.float32)

    if p_vec.numel() != sel_idx.numel():
        raise ValueError(f"missing_rate 长度应为 {sel_idx.numel()}，当前 {p_vec.numel()}")
    if not torch.all((p_vec > 0.0) & (p_vec < 1.0)):
        raise ValueError("missing_rate 需全部在 (0,1) 内。")

    gen = _make_generator(device, seed)

    mask2d = torch.ones(N, D, device=device)  # 默认全观测
    if pattern == "cell":
        # 逐单元格独立伯努利采样：u < p → 缺失
        # 扩展到 [N, C]（C 是被选列数）
        P = p_vec.unsqueeze(0).expand(N, -1)
        U = torch.rand(P.shape, device=device, generator=gen)
        miss = (U < P).float()                     # 1=缺失
        mask_sel = 1.0 - miss                      # 1=观测
        mask2d[:, sel_idx] = mask_sel

    elif pattern == "row":
        # 按时间步（行）采样：u_row < p → 该行在被选列全部缺失
        # 若 p_vec 不是常数，这里采用其均值作为行级 drop 率
        p_row = float(p_vec.mean().clamp(0.0 + 1e-8, 1.0 - 1e-8).cpu())
        Urow = torch.rand((N, 1), device=device, generator=gen)
        keep_row = (Urow >= p_row).float()         # 1=保留行
        mask_sel = keep_row.expand(N, sel_idx.numel())
        mask2d[:, sel_idx] = mask_sel

    else:
        raise ValueError("pattern 只能是 'cell' 或 'row'。")

    # 还原形状并应用
    mask = mask2d.view(B, T, D)
    X_miss = X.clone()
    X_miss[mask == 0] = float("nan")

    # 统计
    info = {
        "pattern": pattern,
        "selected_cols": sel_idx.tolist(),
        "achieved_missing_rate_overall": float((mask == 0).float().mean().cpu()),
        "achieved_missing_rate_selected": float((mask[:, :, sel_idx] == 0).float().mean().cpu()),
    }
    return X_miss


@torch.no_grad()
def generate_rdo_mask(
    X: torch.Tensor,                  # [B, T, D] 完整数据（float）
    row_drop_rate: float = 0.3,       # 行丢失比例（被选列在这些时间步全部缺失），(0,1)
    tail_targets_only: bool = False,  # True: 仅对“最后 f_dim 列”造缺
    f_dim: int | None = None,         # 当 tail_targets_only=True 时必须提供
    mode: str = "bernoulli",          # "bernoulli" | "block"
    share_across_batch: bool = True,  # True: 所有 batch 共享相同行丢失；False: 各样本独立
    mean_off_len: int = 8,            # mode="block" 时，平均缺口长度（时间步）
    seed: int = 42,
):
    """
    返回:
      X_miss: 置缺后的数据 (NaN 表示缺失)，shape=[B,T,D]
      mask:   观测指示 (1=观测, 0=缺失)，shape=[B,T,D]
      info:   元信息（实际缺失率/被造缺列/模式等）

    说明：
      - 这是“整行 drop”：在被选列上，同一时间步要么全观测、要么全缺失。
      - 对被选列的**整体单元格缺失率** ≈ row_drop_rate；
        对全矩阵的整体缺失率 ≈ (len(selected_cols)/D) * row_drop_rate。
    """
    assert X.ndim == 3, "X 必须是 [batch, seq, feature]"
    assert 0 < row_drop_rate < 1, "row_drop_rate 应在 (0,1)"
    device = X.device
    B, T, D = X.shape

    # 选择造缺列
    if tail_targets_only:
        assert f_dim is not None and 1 <= f_dim <= D, "请提供有效的 f_dim"
        sel_idx = torch.arange(D - f_dim, D, device=device)   # 最后 f_dim 列
    else:
        sel_idx = torch.arange(0, D, device=device)           # 全部列
        f_dim = sel_idx.numel()

    gen = _make_generator(device, seed)

    # ---------- 生成“行级 keep 向量” keep_row: 1=保留，0=丢失 ----------
    if mode == "bernoulli":
        # 按行独立伯努利：u < r → 丢失
        if share_across_batch:
            U = torch.rand((T,), device=device, generator=gen)
            keep_row = (U >= row_drop_rate).float().view(1, T, 1).expand(B, T, 1)  # [B,T,1]
        else:
            U = torch.rand((B, T), device=device, generator=gen)
            keep_row = (U >= row_drop_rate).float().unsqueeze(-1)                   # [B,T,1]

    elif mode == "block":
        # 用二状态马尔可夫链生成“开/关”序列（关=丢失，开=观测）
        # 设 π_off = row_drop_rate，平均“关段”长度 E[L_off]=mean_off_len
        # 转移概率： off->on = a = 1/mean_off_len
        #           on->off = b = a * π_off / (1-π_off)
        assert mean_off_len >= 1, "mean_off_len 应 ≥ 1"
        pi_off = float(row_drop_rate)
        a = 1.0 / float(mean_off_len)
        b = a * pi_off / max(1e-8, (1.0 - pi_off))
        a = min(max(a, 1e-6), 1.0)  # 数值裁剪
        b = min(max(b, 1e-6), 1.0)

        def _one_chain_T(T, gen_local):
            # 初始状态按稳态概率 π_off 采样：1=off(丢失), 0=on(观测)
            s = torch.rand((), device=device, generator=gen_local) < pi_off
            out = torch.empty((T,), dtype=torch.float32, device=device)
            for t in range(T):
                out[t] = 0.0 if s == 0 else 1.0   # 记录 off=1（缺失）
                if s == 1:   # off
                    # 以 a 概率转到 on
                    s = 0 if (torch.rand((), device=device, generator=gen_local) < a) else 1
                else:        # on
                    # 以 b 概率转到 off
                    s = 1 if (torch.rand((), device=device, generator=gen_local) < b) else 0
            # keep_row = 1 - off
            return 1.0 - out

        if share_across_batch:
            keep = _one_chain_T(T, gen)
            keep_row = keep.view(1, T, 1).expand(B, T, 1)  # [B,T,1]
        else:
            # 为每个样本独立生成
            keep_list = []
            # （为了复现性，这里用同一个 gen 也能得到确定性；如需更强隔离，可为每个 b 变更种子）
            for _ in range(B):
                keep_list.append(_one_chain_T(T, gen).view(1, T, 1))
            keep_row = torch.cat(keep_list, dim=0)         # [B,T,1]
    else:
        raise ValueError("mode 只能是 'bernoulli' 或 'block'。")

    # ---------- 构造 mask 并应用 ----------
    mask = torch.ones((B, T, D), device=device)
    # 仅对被选列套用 keep_row；非被选列保持观测=1
    mask[:, :, sel_idx] = keep_row.expand(B, T, sel_idx.numel())

    X_miss = X.clone()
    X_miss[mask == 0] = float("nan")

    # 统计
    info = {
        "mode": mode,
        "row_drop_rate_target": row_drop_rate,
        "mean_off_len": mean_off_len if mode == "block" else None,
        "share_across_batch": share_across_batch,
        "selected_cols": sel_idx.tolist(),
        "achieved_missing_rate_overall": float((mask == 0).float().mean().cpu()),
        "achieved_missing_rate_selected": float((mask[:, :, sel_idx] == 0).float().mean().cpu()),
        "achieved_row_drop_rate": float((keep_row == 0).float().mean().cpu()),
    }
    return X_miss
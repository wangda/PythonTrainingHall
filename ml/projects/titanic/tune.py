"""LightGBM 调参实战（核心骨架）：把前置文档 §11.5 的四轮法落成代码，产出调参报告。

用法：
    python tune.py

流程：stage 0（baseline）已实现；stage 1~4 是你要完成的 TODO——
每个 stage 的 docstring 里写了目标、方法和提示，跑通后自动汇总成
config["tune"]["report_path"]（reports/tuning_report.md）。

铁律（前置文档 §11.4）：调参全程只用**训练集** + 分层交叉验证，
测试集留给 train.py / evaluate.py 做最后一次性评估，这里碰都不碰。
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from data.download import load_dataset
from features import fit_preprocessor, split_data

PROJECT_ROOT = Path(__file__).resolve().parent


def make_cv(config):
    """构造分层 K 折交叉验证器（已完整实现）。

    分层（Stratified）保证每折的获救比例和全集一致——类别不平衡数据的标配
    （前置文档 §11.4）。
    """
    from sklearn.model_selection import StratifiedKFold

    return StratifiedKFold(
        n_splits=config["tune"]["n_splits"],
        shuffle=True,
        random_state=config["data"]["random_state"],
    )


def cv_score(params, X, y, cv):
    """一组参数的分层 CV 分数（已完整实现，各 stage 的公共工具）。

    参数:
        params: dict，LGBMClassifier 的构造参数。
        X, y:   训练集特征与标签（注意：只是训练集！）。
        cv:     make_cv 返回的交叉验证器。

    返回:
        (mean, std): float，accuracy 的均值与标准差——报分数永远带 std
                     （前置文档 §11.4 铁律 3：差距小于 std 时那是噪声不是进步）。
    """
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score

    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def stage0_baseline(X, y, config):
    """Stage 0：默认参数基线（已完整实现）。

    不调任何东西，默认参数的 5 折 CV 分数——后续每个 stage 都和它比。
    """
    params = {"random_state": config["data"]["random_state"], "verbose": -1}
    mean, std = cv_score(params, X, y, make_cv(config))
    return {
        "stage": "0 - baseline",
        "what": "默认参数",
        "score": f"{mean:.4f} ± {std:.4f}",
        "best_params": params,
        "observation": "",
    }


def stage1_early_stopping(X, y, config):
    """Stage 1：骨架——learning_rate + 早停定轮数（TODO，前置文档 §11.3/§11.5 第 1 轮）。

    目标:
        固定 learning_rate=0.05、n_estimators 给一个用不完的上限（2000），
        用早停自动找出"该停在哪一轮"，拿到第一轮 CV 分数和最优轮数。

    方法（逐折早停，不能直接用 cross_val_score——早停需要每折内部再切验证集）:
        for train_idx, val_idx in cv.split(X, y):
            1. 取出本折的训练部分 X_fold / y_fold；
            2. 再从训练部分里切一小块早停验证集（train_test_split，
               test_size=config["tune"]["val_size"]，stratify，random_state 固定）；
            3. model.fit(X_inner, y_inner, eval_set=[(X_es, y_es)],
               callbacks=[lgb.early_stopping(config["tune"]["early_stopping_rounds"]),
                          lgb.log_evaluation(0)])
            4. 用 model.best_iteration_ 记录最优轮数，在本折验证部分上算 accuracy；
        汇总：分数取 mean ± std，轮数取中位数（int(np.median(best_iters))）。

    返回:
        dict，与 stage0_baseline 同结构：
            "what": 写明 "lr=0.05 + early_stopping(50)"；
            "best_params": {"learning_rate": 0.05, "n_estimators": 最优轮数中位数, ...}；
            "observation": 留空，报告生成后你手工填写（最优轮数多少？2000 上限够吗？）。

    提示:
        - best_iteration_ 是早停的关键产物——后续 stage 的 n_estimators 都应以它为锚；
        - 逐折手写循环比调 lightgbm.cv 更直白，也能让你看清"每折独立训练"这件事。
    """
    raise NotImplementedError


def stage2_structure(X, y, config, stage1_result):
    """Stage 2：结构——num_leaves × min_child_samples 网格（TODO，前置文档 §11.5 第 2 轮）。

    目标:
        在 stage 1 的骨架上，网格搜索树的复杂度参数：
            num_leaves        ∈ config["tune"]["num_leaves_grid"]（默认 [15, 31, 63]）
            min_child_samples ∈ config["tune"]["min_child_samples_grid"]（默认 [20, 50, 100]）

    方法:
        - 基础参数 = stage1_result["best_params"]（lr + 早停定下的轮数，可直接用
          n_estimators=轮数 跑 cross_val_score，不必再逐折早停——轮数已经定了）；
        - 双重循环 3×3 共 9 组，每组用 cv_score 拿 mean ± std；
        - 选 mean 最高的一组；两组差距 < std 时选**更简单**的（num_leaves 更小）——
          前置文档 §11.4 铁律 3 的应用。

    返回:
        dict："what" 写明网格范围；"best_params" 合并基础参数与胜出组合；
              "observation" 留空（1300 行小数据上，叶子多了有用吗？过拟合了吗？）。

    提示:
        - leaf-wise 生长下 num_leaves 是真正的容量旋钮，max_depth=-1 配合即可（§11.1）；
        - 打印每组网格的分数再选——调参日志的原始素材。
    """
    raise NotImplementedError


def stage3_sampling(X, y, config, stage2_result):
    """Stage 3：采样——feature_fraction × bagging_fraction 网格（TODO，§11.5 第 3 轮）。

    目标:
        在 stage 2 的胜出参数上，网格搜索采样参数（随机森林"去相关"思想在
        boosting 里的移植，前置文档 §9.2/§11.5）：
            feature_fraction  ∈ config["tune"]["feature_fraction_grid"]（默认 4 档）
            bagging_fraction  ∈ config["tune"]["bagging_fraction_grid"]（默认 4 档）

    方法:
        - 注意：bagging_fraction < 1.0 时必须同时给 bagging_freq（如 1），
          否则 LightGBM 不做行采样（静默不生效——静默坑，值得在报告里记一笔）；
        - 16 组网格跑 cv_score，同样遵守"差距 < std 选更简单"的裁决规则；
        - 观察：这份 9~11 个特征的小数据上，特征采样有意义吗？（大概率不如
          结构化数据几百列时明显——这就是"观察与结论"该写的东西。）

    返回:
        dict，同前结构，"best_params" 为截至目前的最优全集。
    """
    raise NotImplementedError


def stage4_regularization(X, y, config, stage3_result):
    """Stage 4：正则 + 收尾（TODO，前置文档 §11.5 第 4 轮）。

    目标:
        1. 在 stage 3 参数上扫 lambda_l2 ∈ config["tune"]["lambda_l2_grid"]
           （默认 [0, 0.1, 1, 10]），选最优；
        2. 收尾动作：learning_rate 降到 config["tune"]["final_learning_rate"]
           （默认 0.02），用 stage 1 的逐折早停重新定轮数——步子变小、树变多，
           通常能白捡一点分数（§10.4  shrinkage 的标准用法）。

    返回:
        dict："best_params" 为最终完整配置；"observation" 留空
              （最终配置 vs stage 0 默认参数，提升多少？这个提升大于 std 吗？）。

    提示:
        - 把最终 best_params 抄回 config.yaml 的 model.params，train.py 就会用它训练；
        - 如果正则网格全都不如不加，诚实写"λ=0 最优"也是结论——调参不是必须调出提升。
    """
    raise NotImplementedError


def write_report(results, config):
    """把各 stage 结果写成 Markdown 调参报告（已完整实现）。

    参数:
        results: list[dict]，各 stage 返回的字典（跳过未实现的）。
        config:  dict，读取 config["tune"]["report_path"]。

    返回:
        Path，报告文件路径。
    """
    report_path = PROJECT_ROOT / config["tune"]["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Titanic LightGBM 调参报告",
        "",
        f"数据：训练集 5 折分层 CV（n_splits={config['tune']['n_splits']}），指标 accuracy。",
        "评分标准：不是最终分数，而是每个 stage 的\"观察与结论\"。",
        "",
        "| Stage | 试了什么 | CV 分数（mean ± std） | 观察与结论 |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['stage']} | {r['what']} | {r['score']} | {r['observation']} |")
    lines += [
        "",
        "## 最终配置",
        "",
        "```yaml",
        yaml.safe_dump({"model": {"name": "lightgbm", "params": results[-1]["best_params"]}},
                       allow_unicode=True).strip(),
        "```",
        "",
        "> 把上面的 params 抄回 config.yaml，重新跑 train.py + evaluate.py。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"调参报告已写入 {report_path}（请补填\"观察与结论\"列）")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Titanic：LightGBM 调参四轮法")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    args = parser.parse_args()

    path = Path(args.config) if args.config else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    try:
        import lightgbm  # noqa: F401
    except ImportError:
        raise SystemExit("tune.py 需要 lightgbm：pip install lightgbm")

    # 铁律：只用训练集调参（前置文档 §11.4）；预处理状态只在训练集上 fit（项目文档 §3）
    df, _ = load_dataset(config)
    train_df, _ = split_data(df, config)
    X, y, _ = fit_preprocessor(train_df, config)
    print(f"调参数据：训练集 {X.shape}（测试集不碰）")

    results = [stage0_baseline(X, y, config)]
    print(f"stage 0: {results[-1]['score']}")

    stages = [stage1_early_stopping, stage2_structure, stage3_sampling, stage4_regularization]
    for i, stage in enumerate(stages, start=1):
        try:
            if i == 1:
                r = stage(X, y, config)
            else:
                r = stage(X, y, config, results[-1])
            results.append(r)
            print(f"stage {i}: {r['score']}  best={r['best_params']}")
        except NotImplementedError:
            print(f"stage {i}: TODO 未实现，跳过（实现后重跑 tune.py 即可续上）")
            break

    write_report(results, config)


if __name__ == "__main__":
    main()

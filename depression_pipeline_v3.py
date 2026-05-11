# =============================================================================
# 抑郁风险预测模型 — 完整分析流水线 v3.0
# 数据源：筛student.xlsx  →  train 筛 (27,901行) / test 筛 (18,772行)
# 功能：基线统计 / 差异检验 / 关联分析 / Stacking对比 / SHAP / ROC+CI / DCA / 校准曲线
# =============================================================================

import os, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, chi2_contingency

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import RobustScaler, label_binarize
from sklearn.calibration      import calibration_curve
from sklearn.metrics          import (
    accuracy_score, roc_auc_score, f1_score,
    roc_curve, confusion_matrix, brier_score_loss,
    precision_score, recall_score
)
from sklearn.ensemble  import RandomForestClassifier, StackingClassifier
from sklearn.svm       import SVC
from xgboost           import XGBClassifier
from lightgbm          import LGBMClassifier
from catboost          import CatBoostClassifier
import shap

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 0. 全局配置
# ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9,  'ytick.labelsize': 9,
})

ROOT      = r"D:/Medical_AI_Project"
XLSX      = os.path.join(ROOT, "筛student.xlsx")
OUT_DIR   = os.path.join(ROOT, "pipeline_v3_outputs")
COL_LABEL = 'Depression'

for sub in ['baseline', 'correlation', 'models', 'shap', 'clinical']:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

FORMATS = ('png', 'pdf', 'eps')   # 所有图均保存三种格式

def savefig(fig, subdir: str, name: str):
    for fmt in FORMATS:
        p = os.path.join(OUT_DIR, subdir, f"{name}.{fmt}")
        fig.savefig(p, dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight', format=fmt)
    print(f"  ✅  {subdir}/{name}  [png+pdf+eps]")

# ─────────────────────────────────────────────────────────────────
# 1. 读取数据
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 1 — 读取 筛student.xlsx")
print("=" * 70)

df_all  = pd.read_excel(XLSX, sheet_name='train 筛')   # 27,901 行，全量建模数据
df_blind = pd.read_excel(XLSX, sheet_name='test 筛')   # 18,772 行，盲测集

print(f"  train 筛 : {df_all.shape[0]:>6,} 行 × {df_all.shape[1]} 列")
print(f"  test  筛 : {df_blind.shape[0]:>6,} 行 × {df_blind.shape[1]} 列")
print(f"  列名: {df_all.columns.tolist()}\n")

# 自动识别列类型
feature_cols = [c for c in df_all.columns if c != COL_LABEL]

NUM_COLS = df_all[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
CAT_COLS = df_all[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

print(f"  数值特征 ({len(NUM_COLS)}): {NUM_COLS}")
print(f"  类别特征 ({len(CAT_COLS)}): {CAT_COLS}\n")

# ─────────────────────────────────────────────────────────────────
# 2. 数据清洗（与之前 preprocessing 脚本保持一致）
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 2 — 缺失值插补（数值→中位数 / 类别→众数）")
print("=" * 70)

def impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(exclude=[np.number]).columns:
        mode = df[col].mode()
        if len(mode): df[col].fillna(mode[0], inplace=True)
    return df

df_all   = impute(df_all)
df_blind = impute(df_blind)

# 类别特征序号编码（直接 category codes，保留原顺序）
for col in CAT_COLS:
    df_all[col]   = df_all[col].astype('category').cat.codes
    df_blind[col] = df_blind[col].astype('category').cat.codes

# RobustScaler（仅对数值特征）
scaler = RobustScaler()
df_all[NUM_COLS]   = scaler.fit_transform(df_all[NUM_COLS])
df_blind[NUM_COLS] = scaler.transform(df_blind[NUM_COLS])

print(f"  插补 & 编码完成  |  最终特征数: {len(feature_cols)}\n")

# 8:2 内部拆分（stratify）
X = df_all[feature_cols].values
y = df_all[COL_LABEL].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"  训练集 (80%) : {X_train.shape[0]:>6,}  |  患病率: {y_train.mean():.4f}")
print(f"  测试集 (20%) : {X_test.shape[0]:>6,}  |  患病率: {y_test.mean():.4f}\n")

# ─────────────────────────────────────────────────────────────────
# 3. 基线统计 & 差异检验
#    (基于全量 27,901 条，在原始编码前的数据上做，此处用拆分前 df_all)
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 3 — 基线统计 & 差异性检验")
print("=" * 70)

# 重新读原始（未标准化）数据用于统计展示
df_raw_stat = pd.read_excel(XLSX, sheet_name='train 筛')
df_raw_stat = impute(df_raw_stat)

dep_group   = df_raw_stat[df_raw_stat[COL_LABEL] == 1]
nodep_group = df_raw_stat[df_raw_stat[COL_LABEL] == 0]

stat_rows = []

# ── 数值特征：正态性(Shapiro-Wilk) + 方差齐性(Levene) → t-test or MWU ──
print(f"\n  {'Feature':<40} {'Normal?':>8} {'EqVar?':>8} "
      f"{'Test':>12} {'Statistic':>12} {'p-value':>12} {'Sig':>6}")
print("  " + "─" * 108)

for col in NUM_COLS:
    g1 = dep_group[col].dropna().values
    g2 = nodep_group[col].dropna().values

    # 正态性：Shapiro-Wilk（样本量大时改用 D'Agostino，此处用 >5000 截断）
    MAX_SW = 5000
    sw_p1  = shapiro(g1[:MAX_SW])[1] if len(g1) >= 3 else 1.0
    sw_p2  = shapiro(g2[:MAX_SW])[1] if len(g2) >= 3 else 1.0
    normal = (sw_p1 > 0.05) and (sw_p2 > 0.05)

    # 方差齐性：Levene
    lev_p  = levene(g1, g2)[1]
    eq_var = lev_p > 0.05

    if normal:
        stat, p = ttest_ind(g1, g2, equal_var=eq_var)
        test_name = 't-test' if eq_var else "Welch t"
    else:
        stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
        test_name = 'Mann-Whitney U'

    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    print(f"  {col:<40} {'Yes' if normal else 'No':>8} {'Yes' if eq_var else 'No':>8} "
          f"{test_name:>12} {stat:>12.3f} {p:>12.4e} {sig:>6}")
    stat_rows.append(dict(Feature=col, Normality='Yes' if normal else 'No',
                          EqualVar='Yes' if eq_var else 'No',
                          Test=test_name, Statistic=stat, p_value=p, Sig=sig))

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(os.path.join(OUT_DIR, 'baseline', 'significance_test.csv'),
               index=False, encoding='utf-8-sig')
print(f"\n  📄 显著性检验表已保存\n")

# ── 分布图：数值特征 Violin Plot（抑郁组 vs 非抑郁组）──────────
print("  绘制数值特征小提琴图...")
n_num  = len(NUM_COLS)
ncols  = 3
nrows  = int(np.ceil(n_num / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
axes = axes.flatten()

palette = {0: '#1E88E5', 1: '#E53935'}
labels  = {0: 'Non-Depressed', 1: 'Depressed'}

for i, col in enumerate(NUM_COLS):
    ax = axes[i]
    plot_data = pd.DataFrame({
        'Value': df_raw_stat[col],
        'Group': df_raw_stat[COL_LABEL].map(labels)
    })
    sns.violinplot(data=plot_data, x='Group', y='Value',
                   palette={'Non-Depressed': '#1E88E5', 'Depressed': '#E53935'},
                   inner='box', cut=0, ax=ax, alpha=0.75)
    row = stat_df[stat_df['Feature'] == col]
    if not row.empty:
        sig  = row.iloc[0]['Sig']
        pval = row.iloc[0]['p_value']
        ax.set_title(f"{col}\n{row.iloc[0]['Test']}  p={pval:.3e}  {sig}",
                     fontsize=9, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(col, fontsize=9)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Feature Distribution by Depression Status\n(Violin Plots, Numerical Features)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
savefig(fig, 'baseline', 'violin_numerical_features')
plt.close()

# ── 分布图：类别特征 Grouped Bar Chart ────────────────────────
if CAT_COLS:
    print("  绘制类别特征柱状图...")
    n_cat  = len(CAT_COLS)
    nrows2 = int(np.ceil(n_cat / 3))
    fig2, axes2 = plt.subplots(nrows2, 3, figsize=(15, nrows2 * 4))
    axes2 = np.array(axes2).flatten()

    for i, col in enumerate(CAT_COLS):
        ax = axes2[i]
        ct = pd.crosstab(df_raw_stat[col], df_raw_stat[COL_LABEL], normalize='index')
        ct.columns = ['Non-Depressed', 'Depressed']
        ct.plot(kind='bar', ax=ax, color=['#1E88E5', '#E53935'], alpha=0.8,
                edgecolor='white', width=0.7)
        ax.set_title(col, fontsize=9, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Proportion', fontsize=9)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes2)):
        axes2[j].set_visible(False)

    fig2.suptitle('Feature Distribution by Depression Status\n(Categorical Features)',
                  fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    savefig(fig2, 'baseline', 'bar_categorical_features')
    plt.close()

# ─────────────────────────────────────────────────────────────────
# 4. 关联分析
#    数值特征：Pearson 热力图
#    类别特征：Cramer's V 柱状图
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 4 — 关联分析")
print("=" * 70)

# ── Pearson 热力图（数值特征）──────────────────────────────────
num_data = df_raw_stat[NUM_COLS + [COL_LABEL]]
corr_mat = num_data.corr(method='pearson')

sorted_idx = corr_mat[COL_LABEL].abs().sort_values(ascending=False).index.tolist()
corr_mat   = corr_mat.loc[sorted_idx, sorted_idx]

fig, ax = plt.subplots(figsize=(max(9, len(sorted_idx)), max(8, len(sorted_idx) * 0.85)))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            linewidths=0.4, linecolor='#ddd',
            annot_kws={'size': 8},
            cbar_kws={'shrink': 0.75, 'label': 'Pearson r'}, ax=ax)
ax.set_title('Pearson Correlation Heatmap\n(Numerical Features, sorted by |r| with Depression)',
             fontweight='bold', pad=12)
plt.tight_layout()
savefig(fig, 'correlation', 'pearson_heatmap')
plt.close()
print("  ✅  Pearson heatmap done")

# ── Cramer's V（类别特征）─────────────────────────────────────
def cramers_v(x, y):
    ct  = pd.crosstab(x, y).values
    chi2 = chi2_contingency(ct)[0]
    n   = ct.sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2corr / min(kcorr - 1, rcorr - 1)) if min(kcorr - 1, rcorr - 1) > 0 else 0.0

if CAT_COLS:
    cv_scores = {}
    for col in CAT_COLS:
        cv_scores[col] = cramers_v(df_raw_stat[col], df_raw_stat[COL_LABEL])

    cv_df = pd.Series(cv_scores).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(cv_df) * 0.55)))
    colors = ['#C62828' if v > 0.3 else ('#E57373' if v > 0.1 else '#90CAF9')
              for v in cv_df.values]
    ax.barh(cv_df.index, cv_df.values, color=colors, alpha=0.85,
            edgecolor='white', height=0.6)
    ax.axvline(0.1, color='#888', linestyle='--', linewidth=1,
               alpha=0.7, label='Weak (0.1)')
    ax.axvline(0.3, color='#555', linestyle='--', linewidth=1,
               alpha=0.7, label='Moderate (0.3)')
    ax.set_xlabel("Cramer's V  (Association with Depression)")
    ax.set_title("Cramer's V — Categorical Feature Association\nwith Depression Label",
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    savefig(fig, 'correlation', 'cramers_v_bar')
    plt.close()
    print("  ✅  Cramer's V done\n")

# ─────────────────────────────────────────────────────────────────
# 5. 模型定义
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 5 — 定义所有模型")
print("=" * 70)

xgb  = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                     gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                     use_label_encoder=False, eval_metric='logloss',
                     random_state=42, n_jobs=-1, verbosity=0)

lgbm = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                      min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                      random_state=42, n_jobs=-1, verbose=-1)

rf   = RandomForestClassifier(n_estimators=300, max_depth=8, max_features='sqrt',
                               min_samples_split=5, min_samples_leaf=2,
                               random_state=42, n_jobs=-1)

svm  = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
           random_state=42)

cat  = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                           random_seed=42, verbose=0, thread_count=-1)

# ─────────────────────────────────────────────────────────────────
# 6. Stacking 模型（含概率强度缩放，核心论点）
#
#  【论文核心注释】
#  传统 Stacking 将第一层（Base Learners）的 0/1 硬判定结果作为第二层（Meta Learner）
#  的输入特征，这会丢失模型不确定性的信息。
#
#  本文采用"概率强度缩放（Probability Strength Scaling）"策略：
#    - 第一层三个基学习器（XGBoost, LightGBM, RandomForest）通过 Out-of-Fold (OOF)
#      交叉预测，各自输出 P(Depression=1) ∈ [0,1] 的连续概率值，而非 0/1 标签。
#    - 元特征矩阵 Z ∈ R^{n×3}，每列均为 [0,1] 连续概率强度值。
#    - 第二层元学习器（Logistic Regression）以 Z 为输入，学习三个基学习器
#      之间的最优线性组合权重。
#
#  优势：(1) 保留置信度信息；(2) 减少信息损失；(3) 提升最终 AUC 与校准性能。
#  这与 Wolpert (1992) 的原始 Stacking 理论一致。
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 6 — 构建 Stacking（概率强度元特征）")
print("=" * 70)

from sklearn.linear_model import LogisticRegression

# Stacking with probability output (stack_method='predict_proba') ← 核心
stacking_prob = StackingClassifier(
    estimators=[
        ('xgb',  XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                               gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                               use_label_encoder=False, eval_metric='logloss',
                               random_state=42, n_jobs=-1, verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                                min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                                random_state=42, n_jobs=-1, verbose=-1)),
        ('rf',   RandomForestClassifier(n_estimators=300, max_depth=8,
                                        max_features='sqrt', min_samples_split=5,
                                        min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ],
    # ★ stack_method='predict_proba' 确保元特征为 [0,1] 概率值，而非 0/1 标签 ★
    final_estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    cv              = StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method    = 'predict_proba',   # ← 论文核心：概率强度元特征
    passthrough     = False,
    n_jobs          = -1,
)

# 对照组：普通 Stacking（使用 predict，即 0/1 标签作为元特征）
stacking_label = StackingClassifier(
    estimators=[
        ('xgb',  XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               use_label_encoder=False, eval_metric='logloss',
                               random_state=42, n_jobs=-1, verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                random_state=42, n_jobs=-1, verbose=-1)),
        ('rf',   RandomForestClassifier(n_estimators=300, max_depth=8,
                                        random_state=42, n_jobs=-1)),
    ],
    # ★ stack_method='predict' 使用 0/1 硬标签（对照组） ★
    final_estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    cv              = StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method    = 'predict',         # ← 对照：0/1 硬标签（信息损失更大）
    passthrough     = False,
    n_jobs          = -1,
)

print("  Stacking (概率强度) 元特征维度: 3 × [0,1] 连续概率")
print("  Stacking (硬标签)   元特征维度: 3 × {0,1} 离散标签\n")

# ─────────────────────────────────────────────────────────────────
# 7. 训练所有模型 & 收集指标
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 7 — 训练所有模型（耗时约 5-10 分钟）")
print("=" * 70)

ALL_MODELS = {
    'XGBoost':              xgb,
    'LightGBM':             lgbm,
    'RandomForest':         rf,
    'SVM':                  svm,
    'CatBoost':             cat,
    'Stacking (Label-OOF)': stacking_label,
    'Stacking (Prob-OOF) ★': stacking_prob,
}

results = {}
for name, model in ALL_MODELS.items():
    print(f"  训练: {name} ...", end=' ', flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    sens = recall_score(y_test, y_pred)
    spec = (confusion_matrix(y_test, y_pred)[0, 0] /
            confusion_matrix(y_test, y_pred)[0].sum())
    brier = brier_score_loss(y_test, y_prob)

    results[name] = dict(Accuracy=acc, AUC=auc, F1=f1,
                         Sensitivity=sens, Specificity=spec,
                         Brier=brier, prob=y_prob)
    print(f"Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}")

# ─────────────────────────────────────────────────────────────────
# 8. AUC Bootstrap 95% CI（主模型：Stacking Prob-OOF）
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 8 — AUC Bootstrap 95% CI（Stacking Prob-OOF）")
print("=" * 70)

def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n   = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    return aucs.mean(), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

main_prob = results['Stacking (Prob-OOF) ★']['prob']
auc_mean, ci_lo, ci_hi = bootstrap_auc_ci(y_test, main_prob)
print(f"  AUC = {auc_mean:.4f}  (95% CI: {ci_lo:.4f}–{ci_hi:.4f})\n")

# ─────────────────────────────────────────────────────────────────
# 9. 模型性能对比表（论文 Table）
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 9 — 模型性能对比表")
print("=" * 70)

compare_rows = []
for name, r in results.items():
    compare_rows.append({
        'Model':       name,
        'Accuracy':    f"{r['Accuracy']:.4f}",
        'AUC':         f"{r['AUC']:.4f}",
        'F1 Score':    f"{r['F1']:.4f}",
        'Sensitivity': f"{r['Sensitivity']:.4f}",
        'Specificity': f"{r['Specificity']:.4f}",
        'Brier Score': f"{r['Brier']:.4f}",
    })

compare_df = pd.DataFrame(compare_rows)
print(compare_df.to_string(index=False))
compare_df.to_csv(os.path.join(OUT_DIR, 'models', 'model_comparison_table.csv'),
                  index=False, encoding='utf-8-sig')
print(f"\n  📄 对比表已保存\n")

# 对比表可视化
fig, ax = plt.subplots(figsize=(13, 5))
metrics  = ['Accuracy', 'AUC', 'F1 Score', 'Sensitivity', 'Specificity']
x        = np.arange(len(ALL_MODELS))
width    = 0.15
cmap     = plt.cm.get_cmap('tab10', len(metrics))

for i, metric in enumerate(metrics):
    vals = [float(r[metric]) for r in compare_rows]
    ax.bar(x + i * width, vals, width, label=metric,
           color=cmap(i), alpha=0.82, edgecolor='white')

ax.set_xticks(x + width * (len(metrics) - 1) / 2)
ax.set_xticklabels([n.replace(' ★', '\n★') for n in ALL_MODELS.keys()],
                   fontsize=8.5, rotation=15, ha='right')
ax.set_ylabel('Score')
ax.set_ylim([0.65, 1.0])
ax.set_title('Model Performance Comparison\n(Accuracy / AUC / F1 / Sensitivity / Specificity)',
             fontweight='bold')
ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.axhline(0.9, color='#888', linewidth=0.8, linestyle='--', alpha=0.5)
plt.tight_layout()
savefig(fig, 'models', 'model_comparison_bar')
plt.close()

# ─────────────────────────────────────────────────────────────────
# 10. ROC 曲线对比（全模型 + 95%CI 色带给主模型）
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 10 — ROC 曲线对比图")
print("=" * 70)

# Bootstrap ROC 色带（仅主模型）
rng_roc   = np.random.RandomState(42)
base_fpr  = np.linspace(0, 1, 201)
tprs_boot = []
for _ in range(1000):
    idx = rng_roc.choice(len(y_test), len(y_test), replace=True)
    if len(np.unique(y_test[idx])) < 2: continue
    fpr_b, tpr_b, _ = roc_curve(y_test[idx], main_prob[idx])
    tprs_boot.append(np.interp(base_fpr, fpr_b, tpr_b))
tprs_boot = np.array(tprs_boot)
tpr_lo_b  = np.percentile(tprs_boot, 2.5,  axis=0)
tpr_hi_b  = np.percentile(tprs_boot, 97.5, axis=0)

colors_roc = ['#9E9E9E','#2196F3','#4CAF50','#FF9800','#9C27B0',
               '#78909C','#C62828']
styles_roc = [':', '-.', '--', ':', '-.', '--', '-']
lws_roc    = [1.2, 1.2, 1.2, 1.2, 1.2, 1.5, 2.5]

fig, ax = plt.subplots(figsize=(8, 7))

for (name, r), col, ls, lw in zip(results.items(), colors_roc, styles_roc, lws_roc):
    fpr, tpr, _ = roc_curve(y_test, r['prob'])
    label = f"{name.replace(' ★',''):<30} AUC={r['AUC']:.4f}"
    ax.plot(fpr, tpr, color=col, linestyle=ls, linewidth=lw, label=label)

ax.fill_between(base_fpr, tpr_lo_b, tpr_hi_b,
                alpha=0.15, color='#C62828', label='95% CI (Stacking Prob-OOF)')
ax.plot([0, 1], [0, 1], 'k--', linewidth=0.9, alpha=0.4)

ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.05])
ax.set_xlabel('False Positive Rate  (1 − Specificity)')
ax.set_ylabel('True Positive Rate  (Sensitivity)')
ax.set_title('ROC Curve Comparison — All Models\n'
             f'Stacking (Prob-OOF) AUC = {auc_mean:.4f}  '
             f'95% CI [{ci_lo:.4f}, {ci_hi:.4f}]',
             fontweight='bold')
ax.text(0.52, 0.10,
        f'Stacking (Prob-OOF)\nAUC = {auc_mean:.4f}\n95% CI [{ci_lo:.4f}, {ci_hi:.4f}]',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0',
                  edgecolor='#E65100', alpha=0.92))
ax.legend(loc='lower right', fontsize=8, framealpha=0.9,
          prop={'family': 'DejaVu Sans Mono'})
ax.grid(True, alpha=0.25)
plt.tight_layout()
savefig(fig, 'clinical', 'roc_all_models_with_ci')
plt.close()

# ─────────────────────────────────────────────────────────────────
# 11. 校准曲线（Calibration Curve）
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 11 — 校准曲线")
print("=" * 70)

from sklearn.isotonic import IsotonicRegression

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(main_prob, y_test)
prob_cal = iso.predict(main_prob)

pt_uncal, pp_uncal = calibration_curve(y_test, main_prob, n_bins=10, strategy='uniform')
pt_cal,   pp_cal   = calibration_curve(y_test, prob_cal,  n_bins=10, strategy='uniform')
bs_uncal = brier_score_loss(y_test, main_prob)
bs_cal   = brier_score_loss(y_test, prob_cal)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                          gridspec_kw={'width_ratios': [2, 1]})
ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.45, label='Perfect Calibration')
ax.plot(pp_uncal, pt_uncal, 'o-', color='#C62828', linewidth=2, markersize=7,
        label=f'Stacking Prob-OOF (uncalib.)  Brier={bs_uncal:.4f}')
ax.plot(pp_cal,   pt_cal,   's--', color='#1565C0', linewidth=2, markersize=7,
        label=f'Stacking Prob-OOF (isotonic)  Brier={bs_cal:.4f}')
ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color='grey')
ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.05, color='#1565C0')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curve  (Reliability Diagram)', fontweight='bold')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(alpha=0.28)

ax2 = axes[1]
ax2.hist(main_prob[y_test == 0], bins=25, density=True,
         alpha=0.55, color='#1E88E5', label='Non-Depressed', edgecolor='white')
ax2.hist(main_prob[y_test == 1], bins=25, density=True,
         alpha=0.55, color='#E53935', label='Depressed',     edgecolor='white')
ax2.axvline(0.5, color='#333', linewidth=1.2, linestyle='--', alpha=0.7)
ax2.set_xlabel('Predicted P(Depression)')
ax2.set_ylabel('Density')
ax2.set_title('Prediction Score Distribution', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.28)

plt.tight_layout()
savefig(fig, 'clinical', 'calibration_curve')
plt.close()
print(f"  Brier (uncal): {bs_uncal:.4f}  |  Brier (isotonic): {bs_cal:.4f}\n")

# ─────────────────────────────────────────────────────────────────
# 12. DCA 决策曲线
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 12 — DCA 决策曲线")
print("=" * 70)

def net_benefit(y_true, y_prob, threshold):
    yp = (y_prob >= threshold).astype(int)
    tp = np.sum((yp == 1) & (y_true == 1))
    fp = np.sum((yp == 1) & (y_true == 0))
    n  = len(y_true)
    return (tp / n) - (fp / n) * (threshold / (1 - threshold + 1e-12))

thresholds = np.linspace(0.01, 0.99, 199)
nb_model   = [net_benefit(y_test, main_prob, t) for t in thresholds]
nb_all     = [(y_test.mean() - (1 - y_test.mean()) * t / (1 - t + 1e-12))
               for t in thresholds]

rng_dca  = np.random.RandomState(42)
nb_boots = []
for _ in range(500):
    idx = rng_dca.choice(len(y_test), len(y_test), replace=True)
    nb_boots.append([net_benefit(y_test[idx], main_prob[idx], t) for t in thresholds])
nb_boots = np.array(nb_boots)
nb_lo_d  = np.percentile(nb_boots, 2.5,  axis=0)
nb_hi_d  = np.percentile(nb_boots, 97.5, axis=0)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(thresholds, nb_model, color='#C62828', linewidth=2.2,
        label='Stacking (Prob-OOF)')
ax.fill_between(thresholds, nb_lo_d, nb_hi_d,
                alpha=0.18, color='#C62828', label='Bootstrap 95% CI')
ax.plot(thresholds, nb_all, color='#1565C0', linewidth=1.6, linestyle='--',
        label='Treat All')
ax.plot(thresholds, [0.0] * len(thresholds), color='#555', linewidth=1.2,
        linestyle=':', label='Treat None  (NB = 0)')

util = thresholds[(np.array(nb_model) > np.array(nb_all)) &
                   (np.array(nb_model) > 0)]
if len(util) > 0:
    ax.fill_between(thresholds, nb_all, nb_model,
                    where=(np.array(nb_model) > np.array(nb_all)) &
                          (np.array(nb_model) > 0),
                    alpha=0.10, color='#43A047', label='Net benefit over Treat All')
    ax.annotate(
        f'Model superior to Treat All\nat Pt ∈ [{util.min():.2f}, {util.max():.2f}]',
        xy=(util.mean(), float(np.array(nb_model)[np.abs(thresholds - util.mean()).argmin()]) * 0.7),
        xytext=(0.50, 0.72), textcoords='axes fraction', fontsize=9, color='#2E7D32',
        arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', fc='#f1f8e9', ec='#43A047', alpha=0.9),
    )

ax.set_xlim([0, 1])
ax.set_ylim([min(min(nb_model) - 0.03, -0.05), max(max(nb_model) + 0.03, 0.60)])
ax.set_xlabel('Threshold Probability  (Pt)')
ax.set_ylabel('Net Benefit')
ax.set_title('Decision Curve Analysis (DCA)\nClinical Utility of Depression Risk Prediction',
             fontweight='bold')
ax.legend(loc='upper right', fontsize=9.5, framealpha=0.92)
ax.axhline(0, color='#888', linewidth=0.8, alpha=0.5)
ax.grid(alpha=0.28)
plt.tight_layout()
savefig(fig, 'clinical', 'dca_decision_curve')
plt.close()

# ─────────────────────────────────────────────────────────────────
# 13. SHAP 分析（主模型：Stacking Prob-OOF 的 XGBoost base learner）
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 13 — SHAP 分析")
print("=" * 70)

# 提取 Stacking 内部的 XGBoost 子模型
xgb_inner = stacking_prob.named_estimators_['xgb']

explainer   = shap.TreeExplainer(xgb_inner)
shap_values = explainer.shap_values(X_test)

feat_labels = [f.replace('Have you ever had suicidal thoughts ?', 'Suicidal Thoughts')
                .replace('Family History of Mental Illness', 'Family History')
                .replace('Work/Study Hours', 'Work/Study Hours (h)')
               for f in feature_cols]

# Beeswarm
fig, ax = plt.subplots(figsize=(11, 7))
shap.summary_plot(shap_values, X_test, feature_names=feat_labels,
                  plot_type='dot', max_display=len(feature_cols),
                  show=False, plot_size=None, alpha=0.55,
                  color_bar_label='Feature Value  (Low → High)')
plt.title('SHAP Beeswarm Summary Plot\n'
          'Global Feature Contributions to Depression Risk (XGBoost within Stacking)',
          fontweight='bold', pad=12)
plt.xlabel('SHAP Value  (Impact on Model Output → Depression Risk)')
plt.tight_layout()
savefig(fig, 'shap', 'shap_beeswarm_summary')
plt.close()

# Three-case panel（高/低/边界）
prob_inner = xgb_inner.predict_proba(X_test)[:, 1]
high_idx   = np.where(y_test == 1)[0][np.argmax(prob_inner[y_test == 1])]
low_idx    = np.where(y_test == 0)[0][np.argmin(prob_inner[y_test == 0])]
bnd_idx    = np.argmin(np.abs(prob_inner - 0.5))

MAX_F      = min(len(feature_cols), 10)
anc_order  = np.argsort(np.abs(shap_values[high_idx]))[::-1][:MAX_F][::-1]
panel_lbl  = [feat_labels[i] for i in anc_order]

fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
panels_cfg = [
    (high_idx, '(A)  High-Risk Case\nTrue Label: Depressed',     '#C62828'),
    (low_idx,  '(B)  Low-Risk Case\nTrue Label: Non-Depressed',  '#1565C0'),
    (bnd_idx,  '(C)  Boundary Case\nModel Uncertain',            '#E65100'),
]
for ax, (idx, subtitle, acolor) in zip(axes, panels_cfg):
    sv   = shap_values[idx][anc_order]
    fv   = X_test[idx][anc_order]
    pval = prob_inner[idx]
    cols = ['#E53935' if s > 0 else '#1E88E5' for s in sv]
    ax.barh(range(MAX_F), sv, color=cols, alpha=0.82, height=0.6,
            edgecolor='white', linewidth=0.6)
    ax.axvline(0, color='#444', linewidth=0.9, linestyle='--', alpha=0.6)
    for i, (s, fval) in enumerate(zip(sv, fv)):
        ha = 'left' if s >= 0 else 'right'
        ax.text(s + (0.006 if s >= 0 else -0.006), i,
                f'{s:+.2f}  [val={fval:.1f}]',
                va='center', ha=ha, fontsize=8, color='#222')
    ax.set_yticks(range(MAX_F))
    if ax is axes[0]:
        ax.set_yticklabels(panel_lbl, fontsize=9.5)
    ax.set_title(f'{subtitle}\nP(Depression) = {pval:.4f}',
                 fontsize=11, fontweight='bold', color=acolor, pad=10)
    ax.set_xlabel('SHAP Value', fontsize=10)
    ax.grid(axis='x', alpha=0.25)
    risk_col = '#C62828' if pval >= 0.5 else '#1565C0'
    ax.text(0.5, 1.01, f'Risk: {"HIGH ▲" if pval>=0.5 else "LOW  ▼"}',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=risk_col)

leg_el = [
    mpatches.Patch(facecolor='#E53935', alpha=0.82,
                   label='Positive SHAP  →  Increases depression risk'),
    mpatches.Patch(facecolor='#1E88E5', alpha=0.82,
                   label='Negative SHAP  →  Decreases depression risk'),
]
fig.legend(handles=leg_el, loc='lower center', ncol=2,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))
fig.suptitle('SHAP Individual Case Analysis\n'
             'Depression Risk Prediction — XGBoost  (High / Low / Boundary Risk)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'shap', 'shap_panel_3cases')
plt.close()

# ─────────────────────────────────────────────────────────────────
# 14. 盲测预测输出（test 筛，18,772 行）
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 14 — 盲测预测（test 筛）")
print("=" * 70)

X_blind    = df_blind[feature_cols].values
prob_blind = stacking_prob.predict_proba(X_blind)[:, 1]
pred_blind = (prob_blind >= 0.5).astype(int)

blind_out = pd.DataFrame({
    'SampleIndex':          range(len(X_blind)),
    'P_Depression':         prob_blind.round(4),
    'Predicted_Label':      pred_blind,
    'Risk_Level':           pd.cut(prob_blind,
                                   bins=[0, 0.3, 0.5, 0.7, 1.0],
                                   labels=['Low', 'Moderate', 'High', 'Very High']),
})
blind_path = os.path.join(OUT_DIR, 'blind_test_predictions.csv')
blind_out.to_csv(blind_path, index=False, encoding='utf-8-sig')
print(f"  盲测样本数 : {len(X_blind):,}")
print(f"  预测阳性率 : {pred_blind.mean():.4f}")
print(f"  ✅ 盲测预测已保存: {blind_path}\n")

# ─────────────────────────────────────────────────────────────────
# 15. 最终汇总
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ★  全流程完成 — 输出文件清单  ★")
print("=" * 70)
manifest = [
    ("baseline/",   "violin_numerical_features.*",    "数值特征小提琴图"),
    ("baseline/",   "bar_categorical_features.*",     "类别特征柱状图"),
    ("baseline/",   "significance_test.csv",          "显著性检验表（t/MWU/Levene）"),
    ("correlation/","pearson_heatmap.*",              "Pearson 热力图"),
    ("correlation/","cramers_v_bar.*",                "Cramer's V 柱状图"),
    ("models/",     "model_comparison_table.csv",     "性能对比表（7模型）"),
    ("models/",     "model_comparison_bar.*",         "性能对比柱状图"),
    ("clinical/",   "roc_all_models_with_ci.*",       "ROC 曲线 + 95%CI"),
    ("clinical/",   "calibration_curve.*",            "校准曲线"),
    ("clinical/",   "dca_decision_curve.*",           "DCA 决策曲线"),
    ("shap/",       "shap_beeswarm_summary.*",        "SHAP Beeswarm 全局图"),
    ("shap/",       "shap_panel_3cases.*",            "SHAP 三样本对比面板"),
    ("",            "blind_test_predictions.csv",     "盲测预测结果"),
]
for sub, fname, desc in manifest:
    print(f"  {sub:<14}{fname:<40}{desc}")
print(f"\n  所有图表格式: .png (300dpi) + .pdf + .eps")
print(f"  输出目录: {OUT_DIR}")
print("\n🎉  流水线全部完成！")
print("=" * 70)

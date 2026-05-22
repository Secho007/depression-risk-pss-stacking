# =============================================================================
# 路线A 修复脚本：重新生成 Fig1 / Fig3 / Fig5 / Suppl.Fig4
# 全部基于 Stacking-v2（XGB+LGBM+RF+Cat，4基学习器，PSS概率强度元特征）
#
# 运行前提：jmir_fix.py 已经跑完，jmir_fix_outputs/ 目录存在
# 运行方式：python fix_figures_routeA.py
# 输出目录：D:/Medical_AI_Project/routeA_fix_outputs/
# =============================================================================

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                              accuracy_score, f1_score, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
})

ROOT    = r"D:/Medical_AI_Project"
XLSX    = os.path.join(ROOT, "筛student.xlsx")
OUT_DIR = os.path.join(ROOT, "routeA_fix_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

COL_LABEL = 'Depression'
FORMATS   = ('png', 'pdf', 'eps')

def savefig(fig, name):
    for fmt in FORMATS:
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{fmt}"),
                    dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight', format=fmt)
    print(f"  ✅  {name}  [png+pdf+eps]")

# =============================================================================
# STEP 1 — 加载数据（与 jmir_fix.py 完全一致）
# =============================================================================
print("=" * 66)
print("  STEP 1 — 加载数据（与 jmir_fix.py 保持一致）")
print("=" * 66)

df_raw = pd.read_excel(XLSX, sheet_name='train 筛')
print(f"  原始: {df_raw.shape[0]:,} 行 × {df_raw.shape[1]} 列")

feature_cols_all = [c for c in df_raw.columns if c != COL_LABEL]
NUM_COLS = df_raw[feature_cols_all].select_dtypes(include=[np.number]).columns.tolist()
CAT_COLS = df_raw[feature_cols_all].select_dtypes(exclude=[np.number]).columns.tolist()

# ── 关键：排除不参与建模的特征 ──────────────────────────────────
# 论文方法节说明：id无预测意义；Work Pressure/Job Satisfaction因群体适配性
# 和VIF≥5被排除；最终保留15个特征
EXCLUDE_COLS = ['id', 'Work Pressure', 'Job Satisfaction',
                'Working Professional or Student', 'Profession', 'Name']

MODEL_FEATURES = [c for c in feature_cols_all if c not in EXCLUDE_COLS]
print(f"\n  全部特征: {len(feature_cols_all)} 个")
print(f"  排除特征: {EXCLUDE_COLS}")
print(f"  建模特征: {len(MODEL_FEATURES)} 个")
print(f"  {MODEL_FEATURES}\n")

# 缺失值插补
for col in df_raw.select_dtypes(include=[np.number]).columns:
    df_raw[col].fillna(df_raw[col].median(), inplace=True)
for col in df_raw.select_dtypes(exclude=[np.number]).columns:
    mode = df_raw[col].mode()
    if len(mode): df_raw[col].fillna(mode[0], inplace=True)

# 保存原始统计用副本（小提琴图/热力图使用，在编码/缩放前）
df_stat = df_raw.copy()

# 类别编码
for col in CAT_COLS:
    df_raw[col] = df_raw[col].astype('category').cat.codes

# RobustScaler（仅数值特征）
num_in_model = [c for c in NUM_COLS if c in MODEL_FEATURES]
scaler = RobustScaler()
df_raw[num_in_model] = scaler.fit_transform(df_raw[num_in_model])

X = df_raw[MODEL_FEATURES].values
y = df_raw[COL_LABEL].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"  训练集: {X_train.shape[0]:,}  测试集: {X_test.shape[0]:,}")
print(f"  患病率 — 训练: {y_train.mean():.4f}  测试: {y_test.mean():.4f}\n")

# SHAP特征标签（规范化显示名称）
FEAT_LABELS = [
    c.replace('Have you ever had suicidal thoughts ?', 'Suicidal Thoughts')
     .replace('Family History of Mental Illness', 'Family History')
     .replace('Work/Study Hours', 'Work/Study Hours (h)')
     .replace('Dietary Habits', 'Dietary Habits')
    for c in MODEL_FEATURES
]

# =============================================================================
# STEP 2 — 定义并训练所有模型（与 jmir_fix.py 同参数）
# =============================================================================
print("=" * 66)
print("  STEP 2 — 定义模型")
print("=" * 66)

# ── 各基学习器 ──────────────────────────────────────────────────
base_xgb = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    use_label_encoder=False, eval_metric='logloss',
    random_state=42, n_jobs=-1, verbosity=0)

base_lgbm = LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbose=-1)

base_rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, max_features='sqrt',
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1)

base_cat = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    random_seed=42, verbose=0, thread_count=-1)

# ── Stacking-v2（PSS，4基学习器）★ 论文核心模型 ★ ────────────
stacking_v2 = StackingClassifier(
    estimators=[
        ('xgb',  base_xgb),
        ('lgbm', base_lgbm),
        ('rf',   base_rf),
        ('cat',  base_cat),
    ],
    final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method='predict_proba',  # ★ 概率强度元特征
    passthrough=False,
    n_jobs=-1,
)

# ── 硬标签对照组（消融实验用）────────────────────────────────
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
        ('cat',  CatBoostClassifier(iterations=500, learning_rate=0.05,
                                     depth=6, random_seed=42, verbose=0)),
    ],
    final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method='predict',    # ← 硬标签对照
    passthrough=False,
    n_jobs=-1,
)

# ── 所有对比模型（用于ROC全图）────────────────────────────────
from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale',
                probability=True, random_state=42)

xgb_solo  = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           use_label_encoder=False, eval_metric='logloss',
                           random_state=42, n_jobs=-1, verbosity=0)
lgbm_solo = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1, verbose=-1)
rf_solo   = RandomForestClassifier(n_estimators=300, max_depth=8, max_features='sqrt',
                                    min_samples_split=5, min_samples_leaf=2,
                                    random_state=42, n_jobs=-1)
cat_solo  = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                random_seed=42, verbose=0, thread_count=-1)

ALL_MODELS = {
    'XGBoost':              xgb_solo,
    'LightGBM':             lgbm_solo,
    'RandomForest':         rf_solo,
    'SVM':                  svm_model,
    'CatBoost':             cat_solo,
    'Stacking (Label-OOF)': stacking_label,
    'Stacking-v2 (PSS) ★':  stacking_v2,
}

# =============================================================================
# STEP 3 — 训练所有模型
# =============================================================================
print("=" * 66)
print("  STEP 3 — 训练所有模型（约 8-12 分钟）")
print("=" * 66)

results = {}
for name, model in ALL_MODELS.items():
    print(f"  训练: {name} ...", end=' ', flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results[name] = {
        'Accuracy':    accuracy_score(y_test, y_pred),
        'AUC':         roc_auc_score(y_test, y_prob),
        'F1':          f1_score(y_test, y_pred),
        'Sensitivity': tp / (tp + fn),
        'Specificity': tn / (tn + fp),
        'Brier':       brier_score_loss(y_test, y_prob),
        'prob':        y_prob,
    }
    r = results[name]
    print(f"Acc={r['Accuracy']:.4f}  AUC={r['AUC']:.4f}  Brier={r['Brier']:.4f}")

print()

# =============================================================================
# STEP 4 — Bootstrap AUC 95%CI（Stacking-v2）
# =============================================================================
print("=" * 66)
print("  STEP 4 — Bootstrap AUC 95%CI（Stacking-v2，n=1000）")
print("=" * 66)

def bootstrap_auc(y_true, y_prob, n=1000, seed=42):
    rng  = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2: continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    return aucs.mean(), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

prob_v2 = results['Stacking-v2 (PSS) ★']['prob']
auc_v2, ci_lo, ci_hi = bootstrap_auc(y_test, prob_v2)
print(f"  Stacking-v2  AUC={auc_v2:.4f}  95%CI [{ci_lo:.4f}, {ci_hi:.4f}]\n")

# 打印全部模型汇总表
print(f"  {'Model':<30} {'Acc':>6} {'AUC':>7} {'F1':>7} "
      f"{'Sens':>7} {'Spec':>7} {'Brier':>7}")
print("  " + "─" * 80)
for name, r in results.items():
    print(f"  {name:<30} {r['Accuracy']:>6.4f} {r['AUC']:>7.4f} "
          f"{r['F1']:>7.4f} {r['Sensitivity']:>7.4f} "
          f"{r['Specificity']:>7.4f} {r['Brier']:>7.4f}")

# =============================================================================
# FIG 1 — ROC曲线全模型对比（★高亮Stacking-v2★）
# =============================================================================
print("\n" + "=" * 66)
print("  FIG 1 — ROC曲线（全模型对比，Stacking-v2高亮）")
print("=" * 66)

# Bootstrap ROC色带（Stacking-v2）
rng_roc  = np.random.RandomState(42)
base_fpr = np.linspace(0, 1, 201)
tprs_b   = []
for _ in range(1000):
    idx = rng_roc.choice(len(y_test), len(y_test), replace=True)
    if len(np.unique(y_test[idx])) < 2: continue
    f, t, _ = roc_curve(y_test[idx], prob_v2[idx])
    tprs_b.append(np.interp(base_fpr, f, t))
tprs_b  = np.array(tprs_b)
tpr_lo  = np.percentile(tprs_b, 2.5,  axis=0)
tpr_hi  = np.percentile(tprs_b, 97.5, axis=0)

# 样式配置（PSS Stacking-v2最后画，覆盖在最上层）
plot_order = [
    ('XGBoost',              '#9E9E9E', ':',   1.2),
    ('LightGBM',             '#2196F3', '-.',  1.2),
    ('RandomForest',         '#4CAF50', '--',  1.2),
    ('SVM',                  '#FF9800', ':',   1.2),
    ('CatBoost',             '#9C27B0', '-.',  1.5),
    ('Stacking (Label-OOF)', '#78909C', '--',  1.5),
    ('Stacking-v2 (PSS) ★',  '#C62828', '-',   2.8),
]

fig, ax = plt.subplots(figsize=(8.5, 7.5))

for name, col, ls, lw in plot_order:
    r   = results[name]
    fpr, tpr, _ = roc_curve(y_test, r['prob'])
    # 特殊处理PSS：加CI色带和标注
    if name == 'Stacking-v2 (PSS) ★':
        ax.fill_between(base_fpr, tpr_lo, tpr_hi,
                        alpha=0.15, color='#C62828',
                        label=f'95% CI  (Stacking-v2 PSS)')
        label = (f"Stacking-v2 (PSS) ★     AUC={r['AUC']:.4f}  "
                 f"95%CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    else:
        disp = name.replace(' (PSS) ★', '')
        label = f"{disp:<28} AUC={r['AUC']:.4f}"
    ax.plot(fpr, tpr, color=col, linestyle=ls, linewidth=lw, label=label)

ax.plot([0, 1], [0, 1], 'k--', linewidth=0.9, alpha=0.35)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.set_xlabel('False Positive Rate  (1 − Specificity)')
ax.set_ylabel('True Positive Rate  (Sensitivity)')
ax.set_title(
    'ROC Curve Comparison — All Models\n'
    f'Stacking-v2 (PSS)  AUC = {auc_v2:.4f}  '
    f'95% CI [{ci_lo:.4f}, {ci_hi:.4f}]',
    fontweight='bold')

# 标注框（右下角）
ax.text(0.52, 0.08,
        f'Stacking-v2 (PSS)\nAUC = {auc_v2:.4f}\n'
        f'95% CI [{ci_lo:.4f}, {ci_hi:.4f}]',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0',
                  edgecolor='#C62828', alpha=0.92))

ax.legend(loc='lower right', fontsize=8.2, framealpha=0.9,
          prop={'family': 'DejaVu Sans Mono'})
ax.grid(True, alpha=0.25)
plt.tight_layout()
savefig(fig, 'fig1_roc_all_models_stackingv2')
plt.close()
print(f"  Fig 1 完成：Stacking-v2 AUC={auc_v2:.4f}")

# =============================================================================
# FIG 3 — SHAP蜂群图（基于CatBoost，只用15个建模特征）
# =============================================================================
print("\n" + "=" * 66)
print("  FIG 3 — SHAP蜂群图（CatBoost base learner，15特征）")
print("=" * 66)

# 从Stacking-v2中提取CatBoost子模型
# （CatBoost是最佳单一基学习器，SHAP解释具有代表性）
cat_inner = stacking_v2.named_estimators_['cat']

print("  计算SHAP值（基于CatBoost within Stacking-v2）...")
explainer   = shap.TreeExplainer(cat_inner)
shap_values = explainer.shap_values(X_test)

# 检查shap_values维度（CatBoost可能返回list）
if isinstance(shap_values, list):
    shap_values = shap_values[1]   # 取阳性类

print(f"  SHAP shape: {shap_values.shape}")
print(f"  特征数: {len(FEAT_LABELS)} → {FEAT_LABELS}")

# 蜂群图
fig, ax = plt.subplots(figsize=(11, 7))
shap.summary_plot(
    shap_values, X_test,
    feature_names=FEAT_LABELS,
    plot_type='dot',
    max_display=len(MODEL_FEATURES),
    show=False,
    plot_size=None,
    alpha=0.55,
    color_bar_label='Feature Value  (Low → High)'
)
plt.title(
    'SHAP Beeswarm Summary Plot\n'
    'Global Feature Contributions to Depression Risk (CatBoost within Stacking-v2)',
    fontweight='bold', pad=12)
plt.xlabel('SHAP Value  (Impact on Model Output → Depression Risk)')
plt.tight_layout()
savefig(fig, 'fig3_shap_beeswarm_stackingv2')
plt.close()

# 三样本个案面板（Suppl.Fig 1）
prob_cat_inner = cat_inner.predict_proba(X_test)[:, 1]
high_idx = np.where(y_test == 1)[0][np.argmax(prob_cat_inner[y_test == 1])]
low_idx  = np.where(y_test == 0)[0][np.argmin(prob_cat_inner[y_test == 0])]
bnd_idx  = np.argmin(np.abs(prob_cat_inner - 0.5))

MAX_F     = min(len(MODEL_FEATURES), 10)
anc_order = np.argsort(np.abs(shap_values[high_idx]))[::-1][:MAX_F][::-1]
panel_lbl = [FEAT_LABELS[i] for i in anc_order]

fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
panels_cfg = [
    (high_idx, '(A)  High-Risk Case\nTrue Label: Depressed',    '#C62828'),
    (low_idx,  '(B)  Low-Risk Case\nTrue Label: Non-Depressed', '#1565C0'),
    (bnd_idx,  '(C)  Boundary Case\nModel Uncertain',           '#E65100'),
]
for ax, (idx, subtitle, acolor) in zip(axes, panels_cfg):
    sv   = shap_values[idx][anc_order]
    fv   = X_test[idx][anc_order]
    pval = prob_cat_inner[idx]
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
fig.suptitle(
    'SHAP Individual Case Analysis\n'
    'Depression Risk Prediction — CatBoost within Stacking-v2  '
    '(High / Low / Boundary Risk)',
    fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'supplfig1_shap_panel_3cases_stackingv2')
plt.close()
print("  Fig 3 & Suppl.Fig 1 完成")

# =============================================================================
# FIG 4 — SHAP单特征依赖图（Single-Feature SHAP Dependence Plots）
#
# 对应论文 Fig. 3 / Fig. 4（根据你文章的编号调整 savefig 里的名字）
#
# 逻辑说明：
#   - 选取 SHAP 贡献最大的 6 个特征，各画一个 dependence scatter plot
#   - X轴 = 该特征的原始值（X_test中的值）
#   - Y轴 = 该特征的SHAP值（对预测概率的边际贡献）
#   - 点颜色 = SHAP值大小（红=正贡献/高风险，绿=负贡献/保护）
#   - 红色虚线 = SHAP=0 分界线，标注"零贡献"阈值点
#   - Lowess平滑曲线显示整体趋势
#   - 逻辑与你文章中的图完全一致
# =============================================================================

print("\n" + "=" * 66)
print("  FIG 4 — SHAP单特征依赖图（Top-6特征）")
print("=" * 66)

from scipy.stats import pearsonr

# ── 1. 计算SHAP值（复用已有的cat_inner和shap_values）──────────────
# 如果你把这段代码放在 FIG 3 之后，shap_values 和 cat_inner 已存在
# 如果单独运行，需要先执行 FIG 3 的代码块获取 cat_inner 和 shap_values

# 选取绝对SHAP贡献最大的6个特征
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top6_idx = np.argsort(mean_abs_shap)[::-1][:6]
top6_labels = [FEAT_LABELS[i] for i in top6_idx]

print(f"  Top-6 特征: {top6_labels}")

# ── 2. Lowess平滑辅助函数 ─────────────────────────────────────────
def lowess_smooth(x, y, frac=0.3):
    """Lowess平滑，优先用statsmodels，回退到numpy多项式拟合"""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sorted_idx = np.argsort(x)
        smoothed = lowess(y[sorted_idx], x[sorted_idx], frac=frac, return_sorted=True)
        return smoothed[:, 0], smoothed[:, 1]
    except ImportError:
        # 回退：用多项式平滑（3次）
        sorted_idx = np.argsort(x)
        xs = x[sorted_idx]
        ys = y[sorted_idx]
        coeffs = np.polyfit(xs, ys, deg=3)
        y_fit = np.polyval(coeffs, xs)
        return xs, y_fit

# ── 3. 绘图 ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 颜色映射：与你文章风格一致（红绿渐变）
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm

for plot_idx, (feat_idx, feat_label) in enumerate(zip(top6_idx, top6_labels)):
    ax = axes[plot_idx]

    x_vals = X_test[:, feat_idx]          # 特征原始值（已RobustScaler）
    y_vals = shap_values[:, feat_idx]     # 该特征的SHAP值

    # 颜色：按SHAP值映射（红=高正贡献，蓝绿=低/负贡献）
    norm = TwoSlopeNorm(vmin=y_vals.min(), vcenter=0, vmax=y_vals.max())
    colors = cm.RdYlGn_r(norm(y_vals))    # 红=高风险，绿=保护

    sc = ax.scatter(x_vals, y_vals,
                    c=y_vals,
                    cmap='RdYlGn_r',
                    norm=norm,
                    alpha=0.35,
                    s=8,
                    linewidths=0,
                    rasterized=True)

    # Lowess趋势线
    try:
        x_smooth, y_smooth = lowess_smooth(x_vals, y_vals, frac=0.25)
        ax.plot(x_smooth, y_smooth,
                color='#8B0000', linewidth=2.0, alpha=0.9,
                label='Lowess curve', zorder=5)
    except Exception:
        pass   # statsmodels未安装时跳过

    # SHAP=0 水平参考线
    ax.axhline(0, color='#aaaaaa', linewidth=1.0, linestyle='--', alpha=0.7)

    # 找零点（SHAP=0 附近特征值）：标注在图上
    if y_vals.min() < 0 < y_vals.max():
        # 用平滑曲线估计零点
        try:
            zero_cross_x = np.interp(0, y_smooth, x_smooth)
            ax.axvline(zero_cross_x, color='#C62828', linewidth=1.3,
                       linestyle='--', alpha=0.75)
            ax.text(zero_cross_x, ax.get_ylim()[1] * 0.85,
                    f' {zero_cross_x:.2f}',
                    color='#C62828', fontsize=9, fontweight='bold',
                    va='top', ha='left')
        except Exception:
            pass

    # 正负贡献区域背景（浅色填充）
    ylim = ax.get_ylim()
    ax.fill_between(ax.get_xlim(),
                    [0, 0], [ylim[1], ylim[1]],
                    alpha=0.04, color='#E53935',
                    label='Positive (↑ risk)')
    ax.fill_between(ax.get_xlim(),
                    [ylim[0], ylim[0]], [0, 0],
                    alpha=0.04, color='#1E88E5',
                    label='Negative (↓ risk)')
    ax.set_ylim(ylim)   # 还原（fill_between会改变ylim）

    # 图例（只在第一个子图）
    if plot_idx == 0:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#8B0000', linewidth=2, label='Lowess curve'),
            mpatches.Patch(facecolor='#FFCDD2', alpha=0.5, label='Positive'),
            mpatches.Patch(facecolor='#BBDEFB', alpha=0.5, label='Negative'),
        ]
        ax.legend(handles=legend_elements, fontsize=8,
                  loc='upper left', framealpha=0.85)

    ax.set_xlabel('Feature Value', fontsize=10)
    ax.set_ylabel('SHAP Value', fontsize=10)
    ax.set_title(feat_label, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.2)

    # 颜色条（只在最后一列子图加）
    if plot_idx in [2, 5]:
        cbar = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap='RdYlGn_r'),
            ax=ax, shrink=0.85, pad=0.02
        )
        cbar.set_label('SHAP\nvalue', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

fig.suptitle(
    'Single-Feature SHAP Dependence Plots\n'
    '(Top-6 Predictors, CatBoost within Stacking-v2)',
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
savefig(fig, 'fig4_shap_dependence_top6')   # ← 根据你论文的图号改这里
plt.close()
print("  ✅  Fig 4 SHAP单特征依赖图完成")



# =============================================================================
# FIG 5 — DCA决策曲线（Stacking-v2）
# =============================================================================
print("\n" + "=" * 66)
print("  FIG 5 — DCA决策曲线（Stacking-v2）")
print("=" * 66)

def net_benefit(y_true, y_prob, threshold):
    yp = (y_prob >= threshold).astype(int)
    tp = np.sum((yp == 1) & (y_true == 1))
    fp = np.sum((yp == 1) & (y_true == 0))
    n  = len(y_true)
    return (tp / n) - (fp / n) * (threshold / (1 - threshold + 1e-12))

thresholds = np.linspace(0.01, 0.99, 199)
nb_model   = np.array([net_benefit(y_test, prob_v2, t) for t in thresholds])
nb_all     = np.array([(y_test.mean() - (1 - y_test.mean()) * t / (1 - t + 1e-12))
                        for t in thresholds])

# Bootstrap CI
rng_dca  = np.random.RandomState(42)
nb_boots = []
for _ in range(1000):
    idx = rng_dca.choice(len(y_test), len(y_test), replace=True)
    nb_boots.append([net_benefit(y_test[idx], prob_v2[idx], t) for t in thresholds])
nb_boots = np.array(nb_boots)
nb_lo_d  = np.percentile(nb_boots, 2.5,  axis=0)
nb_hi_d  = np.percentile(nb_boots, 97.5, axis=0)

# 有效阈值区间
util_mask = (nb_model > nb_all) & (nb_model > 0)
util      = thresholds[util_mask]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(thresholds, nb_model, color='#C62828', linewidth=2.2,
        label='Stacking-v2 (PSS)')
ax.fill_between(thresholds, nb_lo_d, nb_hi_d,
                alpha=0.18, color='#C62828', label='Bootstrap 95% CI')
ax.plot(thresholds, nb_all, color='#1565C0', linewidth=1.6, linestyle='--',
        label='Treat All')
ax.plot(thresholds, [0.0] * len(thresholds), color='#555', linewidth=1.2,
        linestyle=':', label='Treat None  (NB = 0)')

if len(util) > 0:
    ax.fill_between(thresholds, nb_all, nb_model,
                    where=util_mask,
                    alpha=0.10, color='#43A047',
                    label='Net benefit over Treat All')
    pt_lo = util.min()
    pt_hi = util.max()
    mid_t = util.mean()
    mid_nb = float(nb_model[np.abs(thresholds - mid_t).argmin()])
    ax.annotate(
        f'Model superior to Treat All\nat Pt ∈ [{pt_lo:.2f}, {pt_hi:.2f}]',
        xy=(mid_t, mid_nb * 0.7),
        xytext=(0.50, 0.72), textcoords='axes fraction',
        fontsize=9, color='#2E7D32',
        arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', fc='#f1f8e9',
                  ec='#43A047', alpha=0.9),
    )
    print(f"  DCA有效阈值区间: [{pt_lo:.2f}, {pt_hi:.2f}]")

ax.set_xlim([0, 1])
ax.set_ylim([min(nb_model.min() - 0.03, -0.05),
             max(nb_model.max() + 0.03, 0.60)])
ax.set_xlabel('Threshold Probability  (Pt)')
ax.set_ylabel('Net Benefit')
ax.set_title('Decision Curve Analysis (DCA)\nClinical Utility of Depression Risk Prediction',
             fontweight='bold')
ax.legend(loc='upper right', fontsize=9.5, framealpha=0.92)
ax.axhline(0, color='#888', linewidth=0.8, alpha=0.5)
ax.grid(alpha=0.28)
plt.tight_layout()
savefig(fig, 'fig5_dca_stackingv2')
plt.close()
print(f"  Fig 5 完成")

# =============================================================================
# SUPPL.FIG 4 — 5折CV稳定性图（真实数值，CatBoost vs Stacking-v2）
# =============================================================================
print("\n" + "=" * 66)
print("  SUPPL.FIG 4 — 5折CV稳定性（真实数值）")
print("=" * 66)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_cv = {
    'CatBoost':         CatBoostClassifier(iterations=500, learning_rate=0.05,
                                            depth=6, random_seed=42, verbose=0,
                                            thread_count=1),
    'Stacking-v2 (PSS)': stacking_v2,
}

cv_results = {}
for name, model in models_cv.items():
    print(f"  CV: {name} ...", end=' ', flush=True)
    n_jobs_cv = 1 if 'CatBoost' in name and 'Stacking' not in name else -1
    scores = cross_validate(model, X_train, y_train,
                            cv=cv5, scoring='roc_auc',
                            n_jobs=n_jobs_cv, return_train_score=False)
    aucs = scores['test_score']
    cv_results[name] = aucs
    print(f"AUC = {aucs.mean():.4f} ± {aucs.std():.4f}  "
          f"[{aucs.min():.4f}, {aucs.max():.4f}]")

cat_cv  = cv_results['CatBoost']
stk_cv  = cv_results['Stacking-v2 (PSS)']

# 绘图
fig, ax = plt.subplots(figsize=(7, 5.5))
data_cv = {
    'CatBoost\n(best single model)':  cat_cv,
    'Stacking-v2\n(PSS, proposed)':   stk_cv,
}
melted = pd.DataFrame(data_cv).melt(var_name='Model', value_name='AUC (5-Fold CV)')

sns.boxplot(data=melted, x='Model', y='AUC (5-Fold CV)',
            palette={'CatBoost\n(best single model)': '#1E88E5',
                     'Stacking-v2\n(PSS, proposed)':  '#C62828'},
            width=0.45, ax=ax, linewidth=1.5)
sns.stripplot(data=melted, x='Model', y='AUC (5-Fold CV)',
              color='#222', size=8, jitter=False, ax=ax, zorder=5)

for i, (name, vals) in enumerate(data_cv.items()):
    ax.text(i, vals.mean() + 0.0008,
            f'μ = {vals.mean():.4f}\nSD = {vals.std():.4f}',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold',
            color='#1E88E5' if i == 0 else '#C62828')

# 注释文字用真实数值
brier_stk = results['Stacking-v2 (PSS) ★']['Brier']
brier_cat = results['CatBoost']['Brier']
ax.text(0.5, 0.05,
        f'Both models show comparable stability (SD ≈ {cat_cv.std():.4f} / {stk_cv.std():.4f}).\n'
        f'The ensemble advantage lies in calibration (Brier: {brier_stk:.4f} vs {brier_cat:.4f})\n'
        f'and SHAP-based clinical interpretability.',
        transform=ax.transAxes, ha='center', va='bottom',
        fontsize=8.5, style='italic', color='#444',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                  edgecolor='#bbb', alpha=0.9))

ax.set_title('5-Fold Cross-Validation Stability\nCatBoost vs Stacking-v2 (PSS)',
             fontweight='bold', fontsize=12)
ax.set_ylabel('AUC (5-Fold CV)', fontsize=11)
ax.set_xlabel('')
ax.set_ylim([max(0.900, melted['AUC (5-Fold CV)'].min() - 0.005),
             min(1.000, melted['AUC (5-Fold CV)'].max() + 0.012)])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig(fig, 'supplfig4_cv_stability_stackingv2')
plt.close()
print(f"  Suppl.Fig 4 完成")

# =============================================================================
# 最终汇总统计（写入txt，直接用于文字修改）
# =============================================================================
print("\n" + "=" * 66)
print("  ★  数值汇总（用于更新论文文字）★")
print("=" * 66)

r_v2 = results['Stacking-v2 (PSS) ★']
r_cat = results['CatBoost']
r_lbl = results['Stacking (Label-OOF)']

summary = f"""
Stacking-v2 (PSS) — 路线A 最终统计
{'='*55}
测试集 (n={len(y_test):,})
  AUC          : {r_v2['AUC']:.4f}
  AUC 95%CI    : [{ci_lo:.4f}, {ci_hi:.4f}]
  Accuracy     : {r_v2['Accuracy']:.4f}
  Sensitivity  : {r_v2['Sensitivity']:.4f}
  Specificity  : {r_v2['Specificity']:.4f}
  F1 Score     : {r_v2['F1']:.4f}
  Brier Score  : {r_v2['Brier']:.4f}

5-Fold CV（训练集）
  Stacking-v2  : {stk_cv.mean():.4f} ± {stk_cv.std():.4f}
                 [{stk_cv.min():.4f}, {stk_cv.max():.4f}]
  CatBoost     : {cat_cv.mean():.4f} ± {cat_cv.std():.4f}
                 [{cat_cv.min():.4f}, {cat_cv.max():.4f}]

消融实验对比（Stacking-v2 vs Label-OOF）
  AUC提升      : {(r_v2['AUC'] - r_lbl['AUC'])*100:.2f} 个百分点
                 ({r_lbl['AUC']:.4f} → {r_v2['AUC']:.4f})
  Brier改善    : {(r_lbl['Brier'] - r_v2['Brier'])/r_lbl['Brier']*100:.1f}%
                 ({r_lbl['Brier']:.4f} → {r_v2['Brier']:.4f})

DCA有效阈值区间: [{pt_lo:.2f}, {pt_hi:.2f}]  (模型优于Treat All)
"""

print(summary)
with open(os.path.join(OUT_DIR, 'routeA_final_statistics.txt'), 'w',
          encoding='utf-8') as f:
    f.write(summary)

print(f"\n  输出目录: {OUT_DIR}")
print("\n🎉  路线A修复完成！")
print(f"""
  生成文件：
  ├── fig1_roc_all_models_stackingv2.*        → 替换论文 Fig 1
  ├── fig3_shap_beeswarm_stackingv2.*         → 替换论文 Fig 3
  ├── fig5_dca_stackingv2.*                   → 替换论文 Fig 5
  ├── supplfig1_shap_panel_3cases_stackingv2.* → 替换补充图 Suppl.Fig 1
  ├── supplfig4_cv_stability_stackingv2.*     → 替换补充图 Suppl.Fig 4
  └── routeA_final_statistics.txt             → 文字修改依据
""")
print("=" * 66)
# =============================================================================
# FIG 2 — 消融实验图（更新版，与本次训练数值一致）
# =============================================================================
print("\n" + "=" * 66)
print("  FIG 2 — 消融实验图（更新）")
print("=" * 66)

r_v2  = results['Stacking-v2 (PSS) ★']
r_lbl = results['Stacking (Label-OOF)']

auc_v2    = r_v2['AUC']
auc_lbl   = r_lbl['AUC']
brier_v2  = r_v2['Brier']
brier_lbl = r_lbl['Brier']
f1_v2     = r_v2['F1']
f1_lbl    = r_lbl['F1']

auc_gain   = (auc_v2 - auc_lbl) * 100
brier_gain = (brier_lbl - brier_v2) / brier_lbl * 100

fig, axes = plt.subplots(1, 3, figsize=(14, 6))
fig.suptitle(
    'Ablation Study: Probability Strength Scaling (PSS) vs Hard-Label Meta-Features\n'
    'Impact on Stacking Ensemble Performance',
    fontweight='bold', fontsize=13)

configs = [
    ('AUC',
     [auc_lbl, auc_v2],
     f'+{auc_gain:.2f}%',
     min(auc_lbl, auc_v2) - 0.01,
     max(auc_lbl, auc_v2) + 0.01),
    ('F1 Score',
     [f1_lbl, f1_v2],
     f'+{(f1_v2-f1_lbl)/f1_lbl*100:.2f}%',
     min(f1_lbl, f1_v2) - 0.003,
     max(f1_lbl, f1_v2) + 0.003),
    ('Brier Score\n(lower = better)',
     [brier_lbl, brier_v2],
     f'−{brier_gain:.1f}%',
     min(brier_lbl, brier_v2) - 0.003,
     max(brier_lbl, brier_v2) + 0.003),
]

colors = ['#F4913B', '#C0392B']
labels = ['Stacking\n(Hard-label)', 'Stacking-v2\n(PSS)']

for ax, (metric, vals, delta, ylo, yhi) in zip(axes, configs):
    bars = ax.bar(labels, vals, color=colors, width=0.45,
                  edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (yhi - ylo) * 0.01,
                f'{val:.4f}', ha='center', va='bottom',
                fontweight='bold', fontsize=11)
    y_arr = max(vals) + (yhi - ylo) * 0.08
    ax.annotate('', xy=(1, y_arr), xytext=(0, y_arr),
                arrowprops=dict(arrowstyle='<->',
                                color='#1A237E', lw=2.0))
    ax.text(0.5, y_arr + (yhi - ylo) * 0.02, delta,
            ha='center', va='bottom', fontsize=11,
            fontweight='bold',
            color='#1A237E' if 'Brier' not in metric else '#C0392B')
    ax.set_title(metric, fontweight='bold', fontsize=12)
    ax.set_ylim([ylo, yhi + (yhi - ylo) * 0.22])
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
savefig(fig, 'fig2_ablation_updated')
plt.close()
print(f"  ✅ Fig 2 完成")
print(f"     AUC:   {auc_lbl:.4f} → {auc_v2:.4f}  (+{auc_gain:.2f}pp)")
print(f"     Brier: {brier_lbl:.4f} → {brier_v2:.4f}  (−{brier_gain:.1f}%)")

# =============================================================================
# SUPPL. FIG 2 — 小提琴图（连续特征按抑郁状态分组分布）
# =============================================================================
print("\n" + "=" * 66)
print("  SUPPL. FIG 2 — 特征分布小提琴图")
print("=" * 66)

import matplotlib.patches as mpatches

# 连续特征（在df_stat里找数值列，排除id/标签）
_num_feats = [c for c in df_stat.select_dtypes(include=[np.number]).columns
              if c in MODEL_FEATURES]

# 简化标签映射
_label_map = {
    'Age': 'Age', 'Academic Pressure': 'Academic\nPressure',
    'CGPA': 'CGPA', 'Study Satisfaction': 'Study\nSatisfaction',
    'Work/Study Hours': 'Work/Study\nHours', 'Financial Stress': 'Financial\nStress',
}

n_feats = len(_num_feats)
n_cols = 3
n_rows = int(np.ceil(n_feats / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()

palette = {0: '#5BA4CF', 1: '#E07B54'}
group_labels = {0: 'No apparent risk', 1: 'High risk'}

for i, feat in enumerate(_num_feats):
    ax = axes[i]
    grp0 = df_stat[df_stat[COL_LABEL] == 0][feat].dropna().values
    grp1 = df_stat[df_stat[COL_LABEL] == 1][feat].dropna().values

    # 小提琴
    parts = ax.violinplot([grp0, grp1], positions=[0, 1],
                          showmedians=True, showextrema=False)
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(palette.values())[j])
        pc.set_alpha(0.65)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    # 箱线叠加（紧凑）
    ax.boxplot([grp0, grp1], positions=[0, 1],
               widths=0.08, patch_artist=False,
               medianprops=dict(color='black', linewidth=0),
               whiskerprops=dict(linewidth=0),
               capprops=dict(linewidth=0),
               flierprops=dict(marker='', markersize=0))

    # Mann-Whitney U p值
    from scipy.stats import mannwhitneyu
    _, pval = mannwhitneyu(grp0, grp1, alternative='two-sided')
    pstr = 'p < 0.001' if pval < 0.001 else f'p = {pval:.3f}'
    ax.set_title(f'{_label_map.get(feat, feat)}\n{pstr}', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No apparent\nrisk', 'High risk'], fontsize=9)
    ax.set_ylabel('Value', fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 隐藏多余子图
for j in range(n_feats, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Supplementary Fig. 2. Feature Distribution by Depression Status\n(Continuous Features, Violin + Box)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
savefig(fig, 'supplfig2_violin_continuous')
plt.close()
print("  ✅  Suppl. Fig 2 小提琴图完成")


# =============================================================================
# SUPPL. FIG 3 — Pearson相关热力图（数值特征，按|r|与Depression排序）
# =============================================================================
print("\n" + "=" * 66)
print("  SUPPL. FIG 3 — Pearson相关热力图")
print("=" * 66)

_num_cols_stat = [c for c in df_stat.select_dtypes(include=[np.number]).columns
                  if c in MODEL_FEATURES + [COL_LABEL]]

corr_df = df_stat[_num_cols_stat].copy()

# 按与Depression的|r|排序
corr_with_dep = corr_df.corr()[COL_LABEL].abs().sort_values(ascending=False)
ordered_cols = corr_with_dep.index.tolist()
corr_matrix = corr_df[ordered_cols].corr()

# 标签美化
_lmap2 = {
    'Depression': 'Depression', 'Academic Pressure': 'Academic Pressure',
    'Financial Stress': 'Financial Stress', 'Work/Study Hours': 'Work/Study Hours',
    'Study Satisfaction': 'Study Satisfaction', 'Age': 'Age',
    'CGPA': 'CGPA',
}
tick_labels = [_lmap2.get(c, c) for c in ordered_cols]

fig, ax = plt.subplots(figsize=(9, 7))
import matplotlib.colors as mcolors

# 自定义双色渐变：负=蓝，0=白，正=红
cmap = plt.cm.RdBu_r

im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')

n = len(ordered_cols)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(tick_labels, fontsize=9)

# 在每个格子里写r值
for row in range(n):
    for col in range(n):
        val = corr_matrix.values[row, col]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(col, row, f'{val:.2f}', ha='center', va='center',
                fontsize=7.5, color=color)

ax.set_title('Supplementary Fig. 3. Pearson Correlation Heatmap\n'
             '(Continuous features, sorted by |r| with Depression)',
             fontsize=11, fontweight='bold', pad=12)
plt.tight_layout()
savefig(fig, 'supplfig3_pearson_heatmap')
plt.close()
print("  ✅  Suppl. Fig 3 Pearson热力图完成")
print("\n" + "=" * 66)
print("  全部图片生成完毕")
print("=" * 66)

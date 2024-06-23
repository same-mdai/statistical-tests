import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
from sklearn.utils import resample

# 既存の関数（bootstrap_auc, delong_roc_variance, delong_test, mcnemar_test, plot_roc_curve）はそのまま維持
def bootstrap_auc(y_true, y_pred, n_bootstraps=1000, rng_seed=42):
    n_bootstraps = int(n_bootstraps)
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    return bootstrapped_scores

def delong_roc_variance(ground_truth, predictions_one, predictions_two):
    """
    Computes ROC AUC variance for two models' predictions.
    
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    ground_truth = np.array(ground_truth)
    predictions_one = np.array(predictions_one)
    predictions_two = np.array(predictions_two)
    
    # AUC is computed using the trapezoidal rule
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5*(i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1
        return T2

    V_one = compute_midrank(predictions_one)
    V_two = compute_midrank(predictions_two)
    pos = np.array(ground_truth == 1)
    neg = np.array(ground_truth == 0)
    X_one = np.array(predictions_one[pos])
    X_two = np.array(predictions_two[pos])
    Y_one = np.array(predictions_one[neg])
    Y_two = np.array(predictions_two[neg])
    n1 = len(X_one)
    n2 = len(X_two)
    m1 = len(Y_one)
    m2 = len(Y_two)

    def compute_auc(X, Y):
        return (np.sum(np.less.outer(Y, X)) + 0.5 * np.sum(np.equal.outer(Y, X))) / (m1 * n1)

    auc_one = compute_auc(X_one, Y_one)
    auc_two = compute_auc(X_two, Y_two)

    # Compute the components of the covariance matrix
    theta_one = np.sum(np.less.outer(Y_one, X_one)) / (m1 * n1)
    theta_two = np.sum(np.less.outer(Y_two, X_two)) / (m2 * n2)

    S_one = np.cov(np.array([V_one[pos], V_one[neg]]))
    S_two = np.cov(np.array([V_two[pos], V_two[neg]]))

    var_auc_one = (
        S_one[0, 0] / (n1 ** 2)
        + S_one[1, 1] / (m1 ** 2)
        - 2 * S_one[0, 1] / (n1 * m1)
    ) * (n1 + m1) / (4 * (n1 - 1) * (m1 - 1))
    var_auc_two = (
        S_two[0, 0] / (n2 ** 2)
        + S_two[1, 1] / (m2 ** 2)
        - 2 * S_two[0, 1] / (n2 * m2)
    ) * (n2 + m2) / (4 * (n2 - 1) * (m2 - 1))

    # Compute covariance between AUCs
    S = np.cov(np.array([V_one, V_two]))
    cov_auc = (
        S[0, 1] / (n1 * m2)
        + S[1, 0] / (n2 * m1)
        - S[0, 0] / (n1 * m1)
        - S[1, 1] / (n2 * m2)
    ) * (n1 + n2 + m1 + m2) / (8 * (n1 + n2 - 1) * (m1 + m2 - 1))

    var_delta_auc = var_auc_one + var_auc_two - 2 * cov_auc
    return var_delta_auc

def delong_test(y_true, y1_pred, y2_pred):
    auc1 = roc_auc_score(y_true, y1_pred)
    auc2 = roc_auc_score(y_true, y2_pred)
    var = delong_roc_variance(y_true, y1_pred, y2_pred)
    
    z = (auc1 - auc2) / np.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return auc1, auc2, z, p_value

def mcnemar_test(y_true, y1_pred, y2_pred):
    # 2x2の分割表を作成
    table = pd.crosstab(y1_pred == y_true, y2_pred == y_true)
    
    # マクネマー検定を実行
    result = mcnemar(table, exact=False, correction=True)
    
    return result.statistic, result.pvalue

def plot_roc_curve(y_true, y1_pred, y2_pred):
    fpr1, tpr1, _ = roc_curve(y_true, y1_pred)
    fpr2, tpr2, _ = roc_curve(y_true, y2_pred)
    
    fig, ax = plt.subplots()
    ax.plot(fpr1, tpr1, label='Model 1')
    ax.plot(fpr2, tpr2, label='Model 2')
    ax.plot([0, 1], [0, 1], linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend()
    
    return fig

def main():
    st.set_page_config(page_title="高度な統計解析アプリ", page_icon="📊", layout="wide")
    
    st.title('高度な統計解析アプリ 📊')
    st.sidebar.header('設定')

    # ファイルアップロード
    uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロードしてください", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("ファイルが正常にアップロードされました。")
            
            # データプレビュー
            st.subheader("データプレビュー")
            st.dataframe(df.head())
            
            columns = df.columns.tolist()
            
            # 列選択
            col1, col2, col3 = st.columns(3)
            with col1:
                y_true_col = st.selectbox('真の値の列:', columns)
            with col2:
                y1_pred_col = st.selectbox('予測1の列:', columns)
            with col3:
                y2_pred_col = st.selectbox('予測2の列:', columns)
            
            if st.button('解析を実行', key='run_analysis'):
                with st.spinner('解析を実行中...'):
                    y_true = df[y_true_col]
                    y1_pred = df[y1_pred_col]
                    y2_pred = df[y2_pred_col]

                    # DeLong検定
                    auc1, auc2, z_score, delong_p_value = delong_test(y_true, y1_pred, y2_pred)
                    
                    # 結果表示
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader('DeLong検定結果')
                        st.metric(label="AUC (Model 1)", value=f"{auc1:.4f}")
                        st.metric(label="AUC (Model 2)", value=f"{auc2:.4f}")
                        st.metric(label="Z-score", value=f"{z_score:.4f}")
                        st.metric(label="p値", value=f"{delong_p_value:.4f}")
                    
                    with col2:
                        # ROC曲線のプロット
                        st.subheader('ROC曲線')
                        fig = plot_roc_curve(y_true, y1_pred, y2_pred)
                        st.pyplot(fig)

                    # ブートストラップ信頼区間
                    st.subheader('ブートストラップ信頼区間 (95%)')
                    n_bootstraps = 1000
                    with st.spinner('ブートストラップ計算中...'):
                        bootstrapped_auc1 = bootstrap_auc(y_true, y1_pred, n_bootstraps)
                        bootstrapped_auc2 = bootstrap_auc(y_true, y2_pred, n_bootstraps)
                        
                        ci_lower1, ci_upper1 = np.percentile(bootstrapped_auc1, [2.5, 97.5])
                        ci_lower2, ci_upper2 = np.percentile(bootstrapped_auc2, [2.5, 97.5])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Model 1", value=f"[{ci_lower1:.4f}, {ci_upper1:.4f}]")
                        with col2:
                            st.metric(label="Model 2", value=f"[{ci_lower2:.4f}, {ci_upper2:.4f}]")

                    # マクネマー検定
                    st.subheader('マクネマー検定結果')
                    mcnemar_statistic, mcnemar_p_value = mcnemar_test(y_true, y1_pred.round(), y2_pred.round())
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="統計量", value=f"{mcnemar_statistic:.4f}")
                    with col2:
                        st.metric(label="p値", value=f"{mcnemar_p_value:.4f}")

                    # 追加の統計情報
                    st.subheader('追加の統計情報')
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="サンプル数", value=f"{len(y_true)}")
                    with col2:
                        st.metric(label="陽性クラスの割合", value=f"{y_true.mean():.2%}")
                    with col3:
                        correlation = np.corrcoef(y1_pred, y2_pred)[0, 1]
                        st.metric(label="モデル間の相関係数", value=f"{correlation:.4f}")

                st.success('解析が完了しました！')
                
        except Exception as e:
            st.error(f'エラーが発生しました: {str(e)}')
            st.info('CSVファイルの形式を確認し、再度アップロードしてください。')

    st.sidebar.info('注: このアプリケーションは研究目的で使用されることを想定しています。結果の解釈には注意が必要です。')

if __name__ == '__main__':
    main()

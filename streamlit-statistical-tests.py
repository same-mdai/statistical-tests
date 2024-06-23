import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.utils import resample

def preprocess_data(df, y_true_col, y1_pred_col, y2_pred_col):
    """データの前処理を行う関数"""
    y_true =     y1_pred = df[y1_pred_col].astype(float)
    y2_pred = df[y2_pred_col].astype(float)
    
    # 欠損値や無効な値を除外
    mask = ~(np.isnan(y_true) | np.isnan(y1_pred) | np.isnan(y2_pred))
    y_true = y_true[mask]
    y1_pred = y1_pred    y2_pred = y2_pred[mask]
    
    # y_trueを0と1のみに制限
    y_true = (y_true > 0.5).astype(int)
    
    return y_true, y1_pred, y2_pred
def delong_roc_variance(ground_truth, predictions_one, predictions_two):
    """
    Computes variance for AUC estimator for paired ROC curves.
    """
    ground_truth, predictions_one, predictions_two = map(lambda x: np.array(x), (ground_truth, predictions_one, predictions_two))
    assert ground_truth.shape == predictions_one.    
    # Define helper functions
    def auc(y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    def structural_components(y_true, y_pred):
        n = len(y        pos = np.array(y_true == 1)
        neg = np.array(y_true == 0)

        m = len(pos[pos])
        n = len(neg[neg])

        pos_ranks = np.argsort(y_pred[pos])
        neg_ranks = np.arg
        v_pos = np.zeros(m)
        v_neg = np.zeros(n)

        for i in range(m):
            v_pos[i] = np.sum(neg_ranks < pos_ranks        for i in range(n):
            v_neg[i] = np.sum(pos_ranks > neg_ranks[i]) / m

        return v_pos, v_neg

    V_A, V_B = structural_components(ground_truth, predictions_one)
    V_A1, V_B1 = structural_components(ground_truth, predictions_two)

    # Compute the variance
    var_A = np.var(V_A) / len(V_A) + np.var(V_B) / len(V_B)
    var_B = np.var(V_A1) / len(V_A1) + np.var(V_B1) / len(V_B1)
    cov_AB = (np.cov(V_A, V_A1)[0][1] / len(V_A) + np.cov(V_B, V_B1)[0][1] / len(V_B))

    return var_A, var_B, cov_AB

def delong_test(y_true, y1_pred, y2_pred):
    auc1 = roc_auc_score(y_true, y1_pred)
    auc2 = roc_auc_score(y_true, y2_pred)
    var_auc1, var_auc2, cov_auc = delong_roc_variance(y_true, y1_pred, y2_pred)
    
    auc_diff = auc1 - auc2
    var_auc_diff = var_auc1 + var_auc2 - 2 * cov_auc
    z = auc_diff / np.sqrt(var_auc_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return auc1, auc2, z, p_value

def mcnemar_test(y_true, y1_pred, y2_pred):
    y1_binary = (y1_pred > 0.5).astype(int)
    y2_binary = (y2_pred > 0.5).astype(int)
    
    b = np.sum((y1_binary == y_true) & (y2_binary != y_true))
    c = np.sum((y1_binary != y_true) & (y2_binary == y_true))
    
    statistic = (abs(b - c) - 1)**2 / (b + c)
    p_value = stats.chi2.sf(statistic, 1)
    
    return statistic, p_value

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
                    # データの前処理
                    y_true, y1_pred, y2_pred = preprocess_data(df, y_true_col, y1_pred_col, y2_pred_col)
                    
                    if len(y_true) == 0:
                        st.error("有効なデータがありません。データを確認してください。")
                        return

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

                    # マクネマー検定
                    mcnemar_statistic, mcnemar_p_value = mcnemar_test(y_true, y1_pred, y2_pred)
                    
                    st.subheader('マクネマー検定結果')
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

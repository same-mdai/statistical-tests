import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample


def preprocess_data(df, y_true_col, y1_pred_col, y2_pred_col):
    """データの前処理を行う関数"""
    y_true = df[y_true_col].astype(float)
    y1_pred = df[y1_pred_col].astype(float)
    y2_pred = df[y2_pred_col].astype(float)
    
    # 欠損値や無効な値を除外
    mask = ~(np.isnan(y_true) | np.isnan(y1_pred) | np.isnan(y2_pred))
    y_true = y_true[mask]
    y1_pred = y1_pred[mask]
    y2_pred = y2_pred[mask]
    
    # y_trueを0と1のみに制限
    y_true = (y_true > 0.5).astype(int)
    
    return y_true, y1_pred, y2_pred

def delong_roc_variance(ground_truth, predictions_one, predictions_two):
    """
    Computes variance for AUC estimator for paired ROC curves.
    """
    ground_truth, predictions_one, predictions_two = map(lambda x: np.array(x), (ground_truth, predictions_one, predictions_two))
    assert ground_truth.shape == predictions_one.shape == predictions_two.shape
    
    # Define helper functions
    def auc(y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    def structural_components(y_true, y_pred):
        n = len(y_true)
        pos = np.array(y_true == 1)
        neg = np.array(y_true == 0)

        m = len(pos[pos])
        n = len(neg[neg])

        pos_ranks = np.argsort(y_pred[pos])
        neg_ranks = np.argsort(y_pred[neg])

        v_pos = np.zeros(m)
        v_neg = np.zeros(n)

        for i in range(m):
            v_pos[i] = np.sum(neg_ranks < pos_ranks[i]) / n
        for i in range(n):
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

def check_normality(data):
    _, p_value = stats.shapiro(data)
    return p_value > 0.05

def check_homogeneity_of_variance(data1, data2):
    _, p_value = stats.levene(data1, data2)
    return p_value > 0.05

def detect_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    return np.sum((data < lower_bound) | (data > upper_bound))

def plot_distribution(data, title):
    fig, ax = plt.subplots()
    sns.histplot(data, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    return fig

def plot_qq(data, title):
    fig, ax = plt.subplots()
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(title)
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    return fig

def plot_boxplot(data1, data2, labels, title):
    fig, ax = plt.subplots()
    ax.boxplot([data1, data2], labels=labels)
    ax.set_title(title)
    ax.set_xlabel('Groups')
    ax.set_ylabel('Values')
    return fig

def perform_statistical_test(data1, data2, paired=False):
    is_normal1 = check_normality(data1)
    is_normal2 = check_normality(data2)
    is_homogeneous = check_homogeneity_of_variance(data1, data2)
    
    if paired:
        if is_normal1 and is_normal2:
            statistic, p_value = stats.ttest_rel(data1, data2)
            test_name = "対応のあるt検定"
        else:
            statistic, p_value = stats.wilcoxon(data1, data2)
            test_name = "ウィルコクソンの符号順位検定"
    else:
        if is_normal1 and is_normal2 and is_homogeneous:
            statistic, p_value = stats.ttest_ind(data1, data2)
            test_name = "独立サンプルのt検定"
        elif is_normal1 and is_normal2 and not is_homogeneous:
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            test_name = "ウェルチのt検定"
        else:
            statistic, p_value = stats.mannwhitneyu(data1, data2)
            test_name = "マンホイットニーのU検定"
    
    return test_name, statistic, p_value

def main():
    st.set_page_config(page_title="柔軟な統計分析アプリ", page_icon="📊", layout="wide")
    
    st.title('柔軟な統計分析アプリ 📊')
    st.sidebar.header('設定')

    # 分析タイプの選択
    analysis_type = st.sidebar.radio("分析タイプを選択:", ["ROC分析", "統計的検定"])

    # ファイルアップロード
    uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("ファイルが正常にアップロードされました。")
            
            # データプレビュー
            st.subheader("データプレビュー")
            st.dataframe(df.head())
            
            columns = df.columns.tolist()
            
            if analysis_type == "ROC分析":
                # ROC分析用の列選択
                col1, col2, col3 = st.columns(3)
                with col1:
                    y_true_col = st.selectbox('真の値の列:', columns)
                with col2:
                    y1_pred_col = st.selectbox('予測1の列:', columns)
                with col3:
                    y2_pred_col = st.selectbox('予測2の列:', columns)
                
                if st.button('ROC分析を実行', key='run_roc_analysis'):
                    with st.spinner('分析を実行中...'):
                        # データの前処理
                        y_true, y1_pred, y2_pred = preprocess_data(df, y_true_col, y1_pred_col, y2_pred_col)
                        
                        if len(y_true) == 0:
                            st.error("有効なデータがありません。データを確認してください。")
                            return

                        # DeLong検定
                        auc1, auc2, z_score, delong_p_value = delong_test(y_true, y1_pred, y2_pred)
                        
                        # 結果表示
                        st.subheader('DeLong検定結果')
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="AUC (モデル1)", value=f"{auc1:.4f}")
                            st.metric(label="AUC (モデル2)", value=f"{auc2:.4f}")
                        with col2:
                            st.metric(label="Z値", value=f"{z_score:.4f}")
                            st.metric(label="p値", value=f"{delong_p_value:.4f}")
                        
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

            else:  # 統計的検定
                # 統計的検定用の列選択
                col1, col2 = st.columns(2)
                with col1:
                    group1_col = st.selectbox('グループ1の列:', columns)
                with col2:
                    group2_col = st.selectbox('グループ2の列:', columns)
                
                # データの対応関係
                is_paired = st.checkbox("データに対応関係がありますか？（例：同じ被験者の繰り返し測定）")
                
                if st.button('統計的検定を実行', key='run_stat_test'):
                    with st.spinner('分析を実行中...'):
                        # データ抽出
                        group1 = df[group1_col].dropna()
                        group2 = df[group2_col].dropna()

                        # データ分布の確認
                        st.subheader("データ分布の確認")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(plot_distribution(group1, f"{group1_col}の分布"))
                            st.pyplot(plot_qq(group1, f"{group1_col}のQ-Qプロット"))
                        with col2:
                            st.pyplot(plot_distribution(group2, f"{group2_col}の分布"))
                            st.pyplot(plot_qq(group2, f"{group2_col}のQ-Qプロット"))

                        st.pyplot(plot_boxplot(group1, group2, [group1_col, group2_col], "グループの箱ひげ図"))

                        # 正規性の検定
                        is_normal1 = check_normality(group1)
                        is_normal2 = check_normality(group2)
                        st.write(f"{group1_col}の正規性: {'正規分布に従う' if is_normal1 else '正規分布に従わない'}")
                        st.write(f"{group2_col}の正規性: {'正規分布に従う' if is_normal2 else '正規分布に従わない'}")

                        # 等分散性の検定
                        is_homogeneous = check_homogeneity_of_variance(group1, group2)
                        st.write(f"等分散性: {'等分散である' if is_homogeneous else '等分散でない'}")

                        # 外れ値の検出
                        outliers1 = detect_outliers(group1)
                        outliers2 = detect_outliers(group2)
                        st.write(f"{group1_col}の外れ値の数: {outliers1}")
                        st.write(f"{group2_col}の外れ値の数: {outliers2}")

                        # サンプルサイズの確認
                        st.write(f"{group1_col}のサンプルサイズ: {len(group1)}")
                        st.write(f"{group2_col}のサンプルサイズ: {len(group2)}")

                        # 統計的検定の実行
                        test_name, statistic, p_value = perform_statistical_test(group1, group2, paired=is_paired)
                        st.subheader(f"統計的検定結果 ({test_name})")
                        st.write(f"検定統計量: {statistic:.4f}")
                        st.write(f"p値: {p_value:.4f}")

                    st.success('分析が完了しました！')
                
        except Exception as e:
            st.error(f'エラーが発生しました: {str(e)}')
            st.info('CSVファイルの形式を確認し、再度アップロードしてください。')

    st.sidebar.info('注: このアプリケーションは研究目的で使用されることを想定しています。結果の解釈には注意が必要です。')

if __name__ == '__main__':
    main()

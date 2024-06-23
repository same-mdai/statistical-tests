import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
from sklearn.utils import resample

# 既存の関数（bootstrap_auc, delong_roc_variance, delong_test, mcnemar_test, plot_roc_curve）はそのまま維持

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

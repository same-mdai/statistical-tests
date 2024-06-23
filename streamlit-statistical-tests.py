import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.utils import resample
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# R のパッケージをインポート
pROC = importr('pROC')
base = importr('base')
stats_r = importr('stats')

# Python と R のデータフレーム変換を自動化
pandas2ri.activate()

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

def r_delong_test(y_true, y1_pred, y2_pred):
    try:
        # データをRのデータフレームに変換
        df = pd.DataFrame({
            'y_true': y_true,
            'y1_pred': y1_pred,
            'y2_pred': y2_pred
        })
        r_df = pandas2ri.py2rpy(df)

        # R の関数を呼び出してROCオブジェクトを作成
        roc1 = pROC.roc(base.paste('y_true', '~', 'y1_pred'), data=r_df)
        roc2 = pROC.roc(base.paste('y_true', '~', 'y2_pred'), data=r_df)

        # DeLong検定を実行
        result = pROC.roc_test(roc1, roc2, method='delong')

        # 結果を取得
        auc1 = pROC.auc(roc1)[0]
        auc2 = pROC.auc(roc2)[0]
        z = result.rx2('statistic')[0]
        p_value = result.rx2('p.value')[0]

        return auc1, auc2, z, p_value
    except Exception as e:
        st.error(f"DeLong検定の実行中にエラーが発生しました: {str(e)}")
        return None, None, None, None

def r_mcnemar_test(y_true, y1_pred, y2_pred):
    try:
        # 予測を二値化
        y1_binary = (y1_pred > 0.5).astype(int)
        y2_binary = (y2_pred > 0.5).astype(int)
        
        # 2x2の分割表を作成
        table = pd.crosstab(y1_binary == y_true, y2_binary == y_true)
        
        # RのデータフレームとSラスノマトリックスに変換
        r_table = pandas2ri.py2rpy(table)
        r_matrix = base.as_matrix(r_table)
        
        # マクネマー検定を実行
        result = stats_r.mcnemar_test(r_matrix)
        
        # 結果を取得
        statistic = result.rx2('statistic')[0]
        p_value = result.rx2('p.value')[0]
        
        return statistic, p_value
    except Exception as e:
        st.error(f"マクネマー検定の実行中にエラーが発生しました: {str(e)}")
        return None, None

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

                    # DeLong検定 (R版)
                    auc1, auc2, z_score, delong_p_value = r_delong_test(y_true, y1_pred, y2_pred)
                    
                    if auc1 is not None:
                        # 結果表示
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader('DeLong検定結果 (R実装)')
                            st.metric(label="AUC (Model 1)", value=f"{auc1:.4f}")
                            st.metric(label="AUC (Model 2)", value=f"{auc2:.4f}")
                            st.metric(label="Z-score", value=f"{z_score:.4f}")
                            st.metric(label="p値", value=f"{delong_p_value:.4f}")
                        
                        with col2:
                            # ROC曲線のプロット
                            st.subheader('ROC曲線')
                            fig = plot_roc_curve(y_true, y1_pred, y2_pred)
                            st.pyplot(fig)

                        # マクネマー検定 (R版)
                        mcnemar_statistic, mcnemar_p_value = r_mcnemar_test(y_true, y1_pred, y2_pred)
                        
                        if mcnemar_statistic is not None:
                            st.subheader('マクネマー検定結果 (R実装)')
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(label="統計量", value=f"{mcnemar_statistic:.4f}")
                            with col2:
                                st.metric(label="p値", value=f"{mcnemar_p_value:.4f}")
                        else:
                            st.error("マクネマー検定の実行に失敗しました。データを確認してください。")

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
                    else:
                        st.error("DeLong検定の実行に失敗しました。データを確認してください。")

                st.success('解析が完了しました！')
                
        except Exception as e:
            st.error(f'エラーが発生しました: {str(e)}')
            st.info('CSVファイルの形式を確認し、再度アップロードしてください。')

    st.sidebar.info('注: このアプリケーションは研究目的で使用されることを想定しています。結果の解釈には注意が必要です。')

if __name__ == '__main__':
    main()

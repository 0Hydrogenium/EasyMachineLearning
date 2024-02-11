# 示例代码使用 SHAP
import shap
from sklearn.svm import SVR
from sklearn.datasets import make_regression

# 创建一个示例数据集
X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
model = SVR(kernel='linear')
model.fit(X, y)

# 使用SHAP进行解释
# explainer = shap.Explainer(model.de)
# shap_values = explainer.shap_values(X)

explainer = shap.KernelExplainer(model.predict, X)
shap_values = explainer.shap_values(X)

# 打印 SHAP 值
print(shap_values)

# 绘制摘要图
shap.summary_plot(shap_values, X)

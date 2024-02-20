from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from io import StringIO
import sys

# 加载数据集
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

params = {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }

# 创建决策树分类器
dt_classifier = GridSearchCV(DecisionTreeClassifier, params, cv=3, verbose=2)

# 使用StringIO捕获输出
original_stdout = sys.stdout
sys.stdout = StringIO()

# 设置verbose参数为1
dt_classifier.fit(X_train, y_train, verbose=1)

# 恢复stdout
sys.stdout = original_stdout

# 获取捕获的输出
output = sys.stdout

# 打印输出
print("Captured Output:")
print(output)

# 进行预测
y_pred = dt_classifier.predict(X_test)

# 计算准确性
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

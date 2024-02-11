import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


def fuzzy_comprehensive_evaluation_model():
    # 创建模糊变量和模糊集合
    technical_skill = ctrl.Antecedent(np.arange(0, 101, 1), 'technical_skill')
    physical_condition = ctrl.Antecedent(np.arange(0, 101, 1), 'physical_condition')
    mental_toughness = ctrl.Antecedent(np.arange(0, 101, 1), 'mental_toughness')
    opponent_strength = ctrl.Antecedent(np.arange(0, 101, 1), 'opponent_strength')

    performance = ctrl.Consequent(np.arange(0, 101, 1), 'performance')

    # 设定模糊隶属度函数
    technical_skill['low'] = fuzz.trimf(technical_skill.universe, [0, 0, 50])
    technical_skill['medium'] = fuzz.trimf(technical_skill.universe, [0, 50, 100])
    technical_skill['high'] = fuzz.trimf(technical_skill.universe, [50, 100, 100])

    physical_condition['low'] = fuzz.trimf(physical_condition.universe, [0, 0, 50])
    physical_condition['medium'] = fuzz.trimf(physical_condition.universe, [0, 50, 100])
    physical_condition['high'] = fuzz.trimf(physical_condition.universe, [50, 100, 100])

    mental_toughness['low'] = fuzz.trimf(mental_toughness.universe, [0, 0, 50])
    mental_toughness['medium'] = fuzz.trimf(mental_toughness.universe, [0, 50, 100])
    mental_toughness['high'] = fuzz.trimf(mental_toughness.universe, [50, 100, 100])

    opponent_strength['low'] = fuzz.trimf(opponent_strength.universe, [0, 0, 50])
    opponent_strength['medium'] = fuzz.trimf(opponent_strength.universe, [0, 50, 100])
    opponent_strength['high'] = fuzz.trimf(opponent_strength.universe, [50, 100, 100])

    performance['poor'] = fuzz.trimf(performance.universe, [0, 0, 50])
    performance['average'] = fuzz.trimf(performance.universe, [0, 50, 100])
    performance['excellent'] = fuzz.trimf(performance.universe, [50, 100, 100])

    # 设定输出的解模糊方法——质心解模糊方式
    performance.defuzzify_method = 'centroid'

    # 设定规则
    rule1 = ctrl.Rule(
        technical_skill['low'] | physical_condition['low'] | mental_toughness['low'] | opponent_strength['low'],
        performance['poor']
    )
    rule2 = ctrl.Rule(
        technical_skill['medium'] | physical_condition['medium'] | mental_toughness['medium'] | opponent_strength['medium'],
        performance['average']
    )
    rule3 = ctrl.Rule(
        technical_skill['high'] | physical_condition['high'] | mental_toughness['high'] | opponent_strength['high'],
        performance['excellent']
    )

    # 创建控制系统
    performance_evaluation = ctrl.ControlSystem([rule1, rule2, rule3])
    performance_evaluator = ctrl.ControlSystemSimulation(performance_evaluation)

    # 输入数据
    performance_evaluator.input['technical_skill'] = 75
    performance_evaluator.input['physical_condition'] = 80
    performance_evaluator.input['mental_toughness'] = 85
    performance_evaluator.input['opponent_strength'] = 60

    # 计算模糊综合评分
    performance_evaluator.compute()

    # 输出结果
    print("模糊综合评分:", performance_evaluator.output['performance'])

    # 打印模糊集合的可视化图表
    technical_skill.view("technical_skill", sim=performance_evaluator)
    physical_condition.view("physical_condition", sim=performance_evaluator)
    mental_toughness.view("mental_toughness", sim=performance_evaluator)
    opponent_strength.view("opponent_strength", sim=performance_evaluator)
    performance.view("performance", sim=performance_evaluator)

    # Perform sensitivity analyze (to change input value)

    # input_var_1:

    # input_values = np.arange(0, 11, 1)
    # output_values = []
    #
    # for val in input_values:
    #     fuzzy_control_sys_simulation.input["input_var_1"] = val
    #     fuzzy_control_sys_simulation.compute()
    #     output_values.append(fuzzy_control_sys_simulation.output["output_var"])
    #
    # plt.plot(
    #     input_values,
    #     output_values,
    #     label="Sensitivity Analysis"
    # )
    # plt.xlabel("Input Variable 1")
    # plt.ylabel("Output Variable")
    # plt.legend()
    # plt.show()
    #
    # return fuzzy_control_sys_simulation.output["output_var"]

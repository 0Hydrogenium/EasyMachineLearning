def get_values_from_container_class(container):
    return container.x_train, container.y_train, container.x_test, container.y_test, container.hyper_params_optimize


def transform_params_list(params_class, params_list, model=None):
    input_params_keys = []
    input_params_values = []
    inner_value_list = []

    keys = params_class.get_params(model).keys() if model else params_class.get_params().keys()
    for i, param in enumerate(params_list):
        if param in keys:
            input_params_keys.append(param)
            if i != 0:
                input_params_values.append(inner_value_list)
            inner_value_list = []
        else:
            inner_value_list.append(param)
    else:
        input_params_values.append(inner_value_list)
    input_params = dict(zip(input_params_keys, input_params_values))

    for k, v in input_params.items():
        if k in keys:
            value_type = params_class.get_params_type(model)[k] if model else params_class.get_params_type()[k]
            try:
                if value_type == "int":
                    input_params[k] = [int(x) for x in input_params[k]]
                elif value_type == "float":
                    input_params[k] = [float(x) for x in input_params[k]]
                elif value_type == "bool":
                    input_params[k] = [x == "True" for x in input_params[k]]
                elif value_type == "str":
                    input_params[k] = [str(x) for x in input_params[k]]
            except Exception:
                input_params[k] = [str(x) for x in input_params[k]]

    return input_params
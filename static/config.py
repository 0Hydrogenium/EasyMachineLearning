class Config:
    # 随机种子
    RANDOM_STATE = 123

    # 预测图展示的点个数
    DISPLAY_RANGE = 100

    # 绘图颜色组
    COLOR_ITER_NUM = 3

    COLORS = [
        "#ca5353",
        "#c874a5",
        "#b674c8",
        "#8274c8",
        "#748dc8",
        "#74acc8",
        "#74c8b7",
        "#74c88d",
        "#a6c874",
        "#e0e27e",
        "#df9b77",
        "#404040",
        "#999999",
        "#d4d4d4"
    ] * COLOR_ITER_NUM

    COLORS_0 = [
        "#8074C8",
        "#7895C1",
        "#A8CBDF",
        "#992224",
        "#B54764",
        "#E3625D",
        "#EF8B67",
        "#F0C284"
    ] * COLOR_ITER_NUM

    COLORS_1 = [
        "#4A5F7E",
        "#719AAC",
        "#72B063",
        "#94C6CD",
        "#B8DBB3",
        "#E29135"
    ] * COLOR_ITER_NUM

    COLORS_2 = [
        "#4485C7",
        "#D4562E",
        "#DBB428",
        "#682487",
        "#84BA42",
        "#7ABBDB",
        "#A51C36"
    ] * COLOR_ITER_NUM

    COLORS_3 = [
        "#8074C8",
        "#7895C1",
        "#A8CBDF",
        "#F5EBAE",
        "#F0C284",
        "#EF8B67",
        "#E3625D",
        "#B54764"
    ] * COLOR_ITER_NUM

    COLORS_4 = [
        "#979998",
        "#C69287",
        "#E79A90",
        "#EFBC91",
        "#E4CD87",
        "#FAE5BB",
        "#DDDDDF"
    ] * COLOR_ITER_NUM

    COLORS_5 = [
        "#91CCC0",
        "#7FABD1",
        "#F7AC53",
        "#EC6E66",
        "#B5CE4E",
        "#BD7795",
        "#7C7979"
    ] * COLOR_ITER_NUM

    COLORS_6 = [
        "#E9687A",
        "#F58F7A",
        "#FDE2D8",
        "#CFCFD0",
        "#B6B3D6"
    ] * COLOR_ITER_NUM

    JS_0 = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'Welcome to EasyMachineLearning!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""








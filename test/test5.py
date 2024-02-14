import gradio as gr

def download_image():
    # 在这里添加下载图片的逻辑，例如使用URL或其他方法获取图片
    image_path = "path/to/your/image.jpg"  # 替换为实际的图片路径
    return image_path

iface = gr.Interface(
    fn=download_image,
    live=True,
    capture_session=True,
    layout="horizontal",
    components=[
        gr.LinePlot,
        gr.Button()
    ],
    theme="light",
)

iface.launch()

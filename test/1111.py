import gradio as gr


with gr.Blocks() as demo:
    error_box = gr.Textbox(label="Error", visible=False)
    name_box = gr.Textbox(label="Name")
    age_box = gr.Number(label="Age", minimum=0, maximum=100)
    symptoms_box = gr.CheckboxGroup(["Cough", "Fever", "Runny Noise"])
    submit_btn = gr.Button("Submit")

    with gr.Column(visible=False) as output_col:
        diagnosis_box = gr.Textbox(label="Diagnosis")
        patient_summary_box = gr.Textbox(label="Patient Summary")

    def submit(name, age, symptoms):
        if len(name) == 0:
            return {error_box: gr.Textbox(value="Enter name", visible=True)}
        return {
            output_col: gr.Column(visible=True),
            diagnosis_box: "covid" if "Cough" in symptoms else "flu",
            patient_summary_box: f"{name}, {age} y/o"
        }

    submit_btn.click(
        submit,
        [name_box, age_box, symptoms_box],
        [error_box, diagnosis_box, patient_summary_box, output_col]
    )

demo.launch()

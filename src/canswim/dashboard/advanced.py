from loguru import logger
from canswim.model import CanswimModel
import gradio as gr
import duckdb

class AdvancedTab:

    def __init__(self, canswim_model: CanswimModel = None):
        self.canswim_model = canswim_model
        with gr.Row():
            self.queryBox = gr.TextArea(value="select * from forecast")

        with gr.Row():
            self.runBtn = gr.Button(value="Run Query", variant="primary")

        with gr.Row():
            self.queryResult = gr.Dataframe()

        self.runBtn.click(
            fn=self.scan_forecasts,
            inputs=[self.queryBox, ],
            outputs=[self.queryResult],
            queue=False,
        )
        self.queryBox.submit(
            fn=self.scan_forecasts,
            inputs=[self.queryBox, ],
            outputs=[self.queryResult],
            queue=False,
        )

    def scan_forecasts(self, query):
        # only run select queries
        if query.strip().upper().startswith('SELECT'):
          return duckdb.sql(query).df()

import logging
try:
    import streamlit as st
except ImportError:
    st = None

class UIAgent:
    def __init__(self, streamlit_mode=False):
        self.logger = logging.getLogger("UIAgent")
        self.streamlit_mode = streamlit_mode and st is not None
        if self.streamlit_mode:
            self.progress_placeholder = st.empty()
            self.progress = 0
            self.progress_text = "Not started"

    def notify(self, message: str):
        if self.streamlit_mode:
            st.info(message)
        else:
            self.logger.info(f"[UIAgent] {message}")
            print(f"[UI] {message}")

    def report_progress(self, step: str, percent: float):
        msg = f"{step}: {percent:.1f}% complete"
        if self.streamlit_mode:
            self.progress = percent / 100.0
            self.progress_text = f"{step}: {percent:.1f}% complete"
            self.progress_placeholder.progress(self.progress, text=self.progress_text)
        self.logger.info(msg)
        print(msg)

    def finish_progress(self, success=True):
        if self.streamlit_mode:
            label = "Done!" if success else "Failed"
            self.progress_placeholder.progress(1.0, text=label)

    def run(self, status: str):
        self.finish_progress(success=(status.lower() == "success"))
        if self.streamlit_mode:
            if status.lower() == "success":
                st.success("Pipeline completed successfully!")
            else:
                st.error(f"Pipeline failed: {status}")
        self.logger.info(f"Pipeline status: {status}")
        print(f"[UI] Pipeline status: {status}")

"""Value Betting Bot - Streamlit Application."""

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Value Betting Bot",
        page_icon="",
        layout="wide",
    )
    st.title("Value Betting Bot")
    st.write("AI-powered football betting bot - Paper Trading MVP")


if __name__ == "__main__":
    main()

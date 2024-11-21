from chatmodel import *

import solara

SYSTEM = {
    "role": "assistant",
    "content": "Welcome to the EUR-Lex Chatbot! Feel free to ask any questions around rulings by the european union regarding energy law.",
}

QA_instance = RAG(verbose=False)


@solara.component
def Page():
    history = solara.use_reactive([SYSTEM])
    user_text = solara.use_reactive("")
    loading = solara.use_reactive(False)
    selected_index = solara.use_reactive(list(INDEX_MAP.keys())[0])

    def chat():
        input = user_text.value
        if input != "":
            chat_history = list(history.value)
            chat_history.append({"role": "user", "content": input})
            assert isinstance(history.value, list)
            loading.set(True)
            answer = QA_instance.ask_question(input)
            chat_history.append({"role": "assistant", "content": answer})

            input = ""
            history.set(chat_history)
            user_text.set("")
            loading.set(False)

    def clear_history():
        history.set([SYSTEM])
        user_text.set("")
        QA_instance.clear_history()

    solara.use_thread(chat, dependencies=[history, user_text, loading])

    def update_index(val):
        QA_instance.set_index(val)

    with solara.Column(style={"background-color": "white"}):
        solara.Select(
            label="Index / Embedding Selection",
            value=selected_index,
            values=list(INDEX_MAP.keys()),
            on_value=update_index,
        )
        for value in history.value:
            if value["role"] == "system":
                continue

            if value["role"] == "user":
                with solara.Card(
                    style={
                        "background": "#add8e6",
                        "display": "flex",
                        "align-items": "center",
                    }
                ):
                    solara.Markdown(
                        "**User:** " + value["content"],
                        style={"margin-bottom": "-20px"},
                    )

            if value["role"] == "assistant":
                with solara.Card(
                    style={
                        "background": "#90ee90",
                        "display": "flex",
                        "align-items": "center",
                    }
                ):
                    if value["content"].startswith("Helpful answer:"):
                        value["content"] = value["content"].replace(
                            "Helpful answer:", "", 1
                        )
                    if value["content"].startswith("**Helpful answer:**"):
                        value["content"] = value["content"].replace(
                            "**Helpful answer:**", "", 1
                        )
                    solara.Markdown(
                        "**Assistant:** " + value["content"].strip(),
                        style={"margin-bottom": "-20px"},
                    )

        with solara.Card(style={"background": "#add8e6"}):
            with solara.Row(
                style={
                    "background": "#add8e6",
                    "display": "flex",
                    "align-items": "center",
                }
            ):
                solara.InputText(
                    "Your question",
                    value=user_text.value,
                    on_value=user_text.set,
                )
                solara.Button("Ask", on_click=chat)
                solara.Button("Clear History", on_click=clear_history)

        if loading.value:
            solara.ProgressLinear(True)

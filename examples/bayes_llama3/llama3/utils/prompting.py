def llama_chat_prompt(
    messages: list[dict],
):
    """
    Formats messages into a prompt for Meta's Llama model.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<>\n", "\n<>\n\n"
    BOS, EOS = "<s>", ""
    DEFAULT_SYSTEM_PROMPT = """
        You are a helpful, respectful and honest assistant.
        Always answer as helpfully as possible, while being safe."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]
    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()}"
        + f"{E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")
    return "".join(messages_list)

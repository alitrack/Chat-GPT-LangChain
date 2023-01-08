import io
import os
import datetime
import openai
import gradio as gr
import requests

from langchain.agents import load_tools, initialize_agent, get_all_tool_names
from langchain.llms import OpenAI

news_api_key = os.environ["NEWS_API_KEY"]
tmdb_bearer_token = os.environ["TMDB_BEARER_TOKEN"]


def set_openai_api_key(api_key, agent):
    if api_key:
        tool_names = get_all_tool_names()

        os.environ["OPENAI_API_KEY"] = api_key
        llm = OpenAI(model_name="text-davinci-003", temperature=0)
        os.environ["OPENAI_API_KEY"] = ""

        tools = load_tools(tool_names, llm=llm, news_api_key=news_api_key, tmdb_bearer_token=tmdb_bearer_token)
        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
        return agent


def chat(inp, history, agent):
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("inp: " + inp)
    history = history or []
    output = agent.run(inp)
    history.append((inp, output))
    return history, history


def do_html_video_speak(testing):
    headers = {"Authorization": f"Bearer {os.environ['EXHUMAN_API_KEY']}"}
    body = {
        'bot_name': 'Masahiro',
        'bot_response': 'This is a test. This is a test. This is a test. This is a test. This is a test',
        'voice_name': 'Masahiro-EN'
    }
    api_endpoint = "https://api.exh.ai/animations/v1/generate_lipsync"
    res = requests.post(api_endpoint, json=body, headers=headers)

    html_video = '<pre>no video</pre>'
    if isinstance(res.content, bytes):
        response_stream = io.BytesIO(res.content)
        with open('videos/tempfile.mp4', 'wb') as f:
            f.write(response_stream.read())
        temp_file = gr.File("videos/tempfile.mp4")
        temp_file_url = "/file=" + temp_file.value['name']
        html_video = f'<video width="256" height="256" autoplay><source src={temp_file_url} type="video/mp4"></video>'
    else:
        print('video url unknown')
    return html_video, "videos/tempfile.mp4"


block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        my_file = gr.File(label="Upload a file", type="file", visible=False)

        tmp_file = gr.File("videos/Masahiro.mp4", visible=False)
        tmp_file_url = "/file=" + tmp_file.value['name']
        htm_video = f'<video width="256" height="256" autoplay muted loop><source src={tmp_file_url} type="video/mp4" poster="Masahiro.png"></video>'
        video_html = gr.HTML(htm_video)

        # video_html = gr.HTML("f'<video width=\"512\" height=\"512\" autoplay><source src={tmp_file_url} type=\"video/mp4\"></video>")

        # video_html = gr.HTML("""<video width="100%" height="100%" controls autoplay loop>
        # <source src="videos/Masahiro.mp4"
        # type="video/mp4"></video>""")

        gr.Markdown("<h3><center>LangChain AI</center></h3>")

        openai_api_key_textbox = gr.Textbox(placeholder="Paste your OpenAI API key (sk-...)",
               show_label=False, lines=1, type='password')

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(label="What's your question?",
                             placeholder="What's the answer to life, the universe, and everything?",
                             lines=1)
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)
        test = gr.Button(value="Test", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=["How many people live in Canada?",
                  "What is 13**.3?",
                  "How much did it rain in SF today?",
                  "Get me information about the movie 'Avatar'",
                  "What are the top tech headlines in the US?",
                  "On the desk, you see two blue booklets, two purple booklets, and two yellow pairs of sunglasses - "
                  "if I remove all the pairs of sunglasses from the desk, how many purple items remain on it?"],
        inputs=message
    )

    gr.HTML("""
    This simple application demonstrates some capabilities of GPT-3 in conjunction with the open source
    LangChain library. It consists of an agent (backed by a GPT-3 language model) using tools to 
    answer/execute questions. It is based upon
     <a href="https://colab.research.google.com/drive/1ZiU0zU17FeLWKkRbxB6AB_NfqvenJkGF"> this Jupyter notebook</a>.""")

    gr.HTML("<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>")

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])
    test.click(do_html_video_speak, inputs=[message], outputs=[video_html, my_file])

    openai_api_key_textbox.change(set_openai_api_key,
                                  inputs=[openai_api_key_textbox, agent_state],
                                  outputs=[agent_state])

block.launch(debug = True)


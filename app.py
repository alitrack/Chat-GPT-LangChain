import io
import os
from typing import Optional, Tuple
import datetime
import gradio as gr
import requests
from langchain import ConversationChain

from langchain.agents import load_tools, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import OpenAI

news_api_key = os.environ["NEWS_API_KEY"]
tmdb_bearer_token = os.environ["TMDB_BEARER_TOKEN"]


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    tool_names = ['serpapi', 'pal-math', 'pal-colored-objects', 'news-api', 'tmdb-api', 'open-meteo-api']

    memory = ConversationBufferMemory(memory_key="chat_history")

    tools = load_tools(tool_names, llm=llm, news_api_key=news_api_key, tmdb_bearer_token=tmdb_bearer_token)
    chain = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=memory)
    return chain


def set_openai_api_key(api_key, agent):
    """Set the api key and return chain.

    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        chain = load_chain()
        os.environ["OPENAI_API_KEY"] = ""
        return chain


def chat(
    inp: str, history: Optional[Tuple[str, str]], chain: Optional[ConversationChain]
):
    """Execute the chat functionality."""
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("inp: " + inp)
    history = history or []
    # If chain is None, that is because no API key was provided.
    output = "Please paste your OpenAI key to use this application."
    if chain and chain != "":
        # Run chain and append input.
        output = chain.run(input=inp)
    history.append((inp, output))
    html_video, temp_file = do_html_video_speak(output)
    return history, history, html_video, temp_file, ""


def do_html_video_speak(words_to_speak):
    headers = {"Authorization": f"Bearer {os.environ['EXHUMAN_API_KEY']}"}
    body = {
        'bot_name': 'Masahiro',
        'bot_response': words_to_speak,
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
        html_video = f'<video width="256" height="256" autoplay><source src={temp_file_url} type="video/mp4" poster="Masahiro.png"></video>'
    else:
        print('video url unknown')
    return html_video, "videos/tempfile.mp4"


block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        with gr.Column():
            gr.Markdown("<h4><center>Conversational Agent using GPT-3.5 & LangChain</center></h4>")

        openai_api_key_textbox = gr.Textbox(placeholder="Paste your OpenAI API key (sk-...)",
               show_label=False, lines=1, type='password')

    with gr.Row():
        with gr.Column(scale=0.25, min_width=240):
            my_file = gr.File(label="Upload a file", type="file", visible=False)
            tmp_file = gr.File("videos/Masahiro.mp4", visible=False)
            tmp_file_url = "/file=" + tmp_file.value['name']
            htm_video = f'<video width="256" height="256" autoplay muted loop><source src={tmp_file_url} type="video/mp4" poster="Masahiro.png"></video>'
            video_html = gr.HTML(htm_video)

        with gr.Column(scale=0.75):
            chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(label="What's your question?",
                             placeholder="What's the answer to life, the universe, and everything?",
                             lines=1)
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=["How many people live in Canada?",
                  "What is 2 to the 30th power?",
                  "How much did it rain in SF today?",
                  "Get me information about the movie 'Avatar'",
                  "What are the top tech headlines in the US?",
                  "On the desk, you see two blue booklets, two purple booklets, and two yellow pairs of sunglasses - "
                  "if I remove all the pairs of sunglasses from the desk, how many purple items remain on it?"],
        inputs=message
    )

    gr.HTML("""
    This simple application demonstrates a conversational agent implemented with OpenAI GPT-3.5 and LangChain. 
    When necessary, it leverages tools for complex math, searching the internet, and accessing news and weather.
    On a desktop, the agent will often speak using using an animated avatar from 
    <a href='https://exh.ai/'>Ex-Human</a>.""")

    gr.HTML("<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>")

    state = gr.State()
    chain_state = gr.State()

    message.submit(chat, inputs=[message, state, chain_state], outputs=[chatbot, state, video_html, my_file, message])
    submit.click(chat, inputs=[message, state, chain_state], outputs=[chatbot, state, video_html, my_file, message])

    openai_api_key_textbox.change(set_openai_api_key,
                                  inputs=[openai_api_key_textbox, chain_state],
                                  outputs=[chain_state])

block.launch(debug = True)


import io
import os
from typing import Optional, Tuple
import datetime
import gradio as gr
import requests

# UNCOMMENT TO USE WHISPER
# import warnings
# import whisper

from langchain import ConversationChain

from langchain.agents import load_tools, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import OpenAI

# Console to variable
from io import StringIO
import sys
import re

from openai.error import AuthenticationError, InvalidRequestError

news_api_key = os.environ["NEWS_API_KEY"]
tmdb_bearer_token = os.environ["TMDB_BEARER_TOKEN"]

TOOLS_LIST = ['serpapi', 'wolfram-alpha', 'google-search', 'pal-math', 'pal-colored-objects', 'news-api', 'tmdb-api',
              'open-meteo-api']
TOOLS_DEFAULT_LIST = ['serpapi', 'pal-math', 'pal-colored-objects']
BUG_FOUND_MSG = "Congratulations, you've found a bug in this application!"
AUTH_ERR_MSG = "Please paste your OpenAI key."


# UNCOMMENT TO USE WHISPER
# warnings.filterwarnings("ignore")
# WHISPER_MODEL = whisper.load_model("tiny")
# print("WHISPER_MODEL", WHISPER_MODEL)


# UNCOMMENT TO USE WHISPER
# def transcribe(aud_inp):
#     if aud_inp is None:
#         return ""
#     aud = whisper.load_audio(aud_inp)
#     aud = whisper.pad_or_trim(aud)
#     mel = whisper.log_mel_spectrogram(aud).to(WHISPER_MODEL.device)
#     _, probs = WHISPER_MODEL.detect_language(mel)
#     options = whisper.DecodingOptions()
#     result = whisper.decode(WHISPER_MODEL, mel, options)
#     print("result.text", result.text)
#     result_text = ""
#     if result and result.text:
#         result_text = result.text
#     return result_text


def load_chain(tools_list, llm):
    print("tools_list", tools_list)
    tool_names = tools_list
    tools = load_tools(tool_names, llm=llm, news_api_key=news_api_key, tmdb_bearer_token=tmdb_bearer_token)

    memory = ConversationBufferMemory(memory_key="chat_history")
    chain = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=memory)
    return chain


def set_openai_api_key(api_key):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key:
        llm = OpenAI(temperature=0, openai_api_key=api_key)
        chain = load_chain(TOOLS_DEFAULT_LIST, llm)
        return chain, llm


def run_chain(chain, inp, capture_hidden_text):
    output = ""
    hidden_text = None
    if capture_hidden_text:
        error_msg = None
        tmp = sys.stdout
        hidden_text_io = StringIO()
        sys.stdout = hidden_text_io

        try:
            output = chain.run(input=inp)
        except AuthenticationError as ae:
            error_msg = AUTH_ERR_MSG
        except ValueError as ve:
            error_msg = "\n\nValueError, " + str(ve)
        except InvalidRequestError as ire:
            error_msg = "\n\n" + BUG_FOUND_MSG + " Here are the details: InvalidRequestError, " + str(ire)
        except Exception as e:
            error_msg = "\n" + BUG_FOUND_MSG + " Here are the details: Exception, " + str(e)

        sys.stdout = tmp
        hidden_text = hidden_text_io.getvalue()

        # remove escape characters from hidden_text
        hidden_text = re.sub(r'\x1b[^m]*m', '', hidden_text)

        # remove "Entering new AgentExecutor chain..." from hidden_text
        hidden_text = re.sub(r"Entering new AgentExecutor chain...\n", "", hidden_text)

        # remove "Finished chain." from hidden_text
        hidden_text = re.sub(r"Finished chain.", "", hidden_text)

        # Add newline after "Thought:" "Action:" "Observation:" "Input:" and "AI:"
        hidden_text = re.sub(r"Thought:", "\n\nThought:", hidden_text)
        hidden_text = re.sub(r"Action:", "\n\nAction:", hidden_text)
        hidden_text = re.sub(r"Observation:", "\n\nObservation:", hidden_text)
        hidden_text = re.sub(r"Input:", "\n\nInput:", hidden_text)
        hidden_text = re.sub(r"AI:", "\n\nAI:", hidden_text)

        if error_msg:
            hidden_text += error_msg

        print("hidden_text: ", hidden_text)
    else:
        try:
            output = chain.run(input=inp)
        except AuthenticationError as ae:
            output = AUTH_ERR_MSG
            print("\nAuthenticationError: ", ae)
        except ValueError as ve:
            output = "ValueError, " + str(ve)
            print("\nValueError: ", ve)
        except InvalidRequestError as ire:
            output = BUG_FOUND_MSG + " Here are the details: InvalidRequestError, " + str(ire)
            print("\nInvalidRequestError: ", ire)
        except Exception as e:
            output = BUG_FOUND_MSG + " Here are the details: Exception, " + str(e)
            print("\nException: ", e)

    return output, hidden_text


def chat(
        inp: str, history: Optional[Tuple[str, str]], chain: Optional[ConversationChain],
        trace_chain: bool):
    """Execute the chat functionality."""
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("inp: " + inp)
    print("trace_chain: ", trace_chain)
    history = history or []
    # If chain is None, that is because no API key was provided.
    output = "Please paste your OpenAI key to use this application."
    hidden_text = output

    if chain and chain != "":
        output, hidden_text = run_chain(chain, inp, capture_hidden_text=trace_chain)

    history.append((inp, hidden_text if trace_chain else output))

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


def update_selected_tools(widget, state, llm):
    if widget:
        state = widget
        chain = load_chain(state, llm)
        return state, llm, chain


def update_foo(widget, state):
    if widget:
        state = widget
        return state


block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    llm_state = gr.State()
    history_state = gr.State()
    chain_state = gr.State()
    tools_list_state = gr.State(TOOLS_DEFAULT_LIST)
    trace_chain_state = gr.State(False)

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

            trace_chain_cb = gr.Checkbox(label="Show chain", value=False)
            trace_chain_cb.change(update_foo, inputs=[trace_chain_cb, trace_chain_state],
                                  outputs=[trace_chain_state])

        with gr.Column(scale=0.75):
            chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(label="What's on your mind??",
                             placeholder="What's the answer to life, the universe, and everything?",
                             lines=1)
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    # UNCOMMENT TO USE WHISPER
    # with gr.Row():
    #     audio_comp = gr.Microphone(source="microphone", type="filepath", label="Just say it!",
    #                                interactive=True, streaming=False)
    #     audio_comp.change(transcribe, inputs=[audio_comp], outputs=[message])

    with gr.Row():
        tools_cb_group = gr.CheckboxGroup(label="Tools:", choices=TOOLS_LIST,
                                          value=TOOLS_DEFAULT_LIST)

        tools_cb_group.change(update_selected_tools,
                              inputs=[tools_cb_group, tools_list_state, llm_state],
                              outputs=[tools_list_state, llm_state, chain_state])

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
    This application demonstrates a conversational agent implemented with OpenAI GPT-3.5 and LangChain. 
    When necessary, it leverages tools for complex math, searching the internet, and accessing news and weather.
    On a desktop, the agent will often speak using using an animated avatar from 
    <a href='https://exh.ai/'>Ex-Human</a>.""")

    gr.HTML("<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>")

    message.submit(chat, inputs=[message, history_state, chain_state, trace_chain_state],
                   outputs=[chatbot, history_state, video_html, my_file, message])
    submit.click(chat, inputs=[message, history_state, chain_state, trace_chain_state],
                 outputs=[chatbot, history_state, video_html, my_file, message])

    openai_api_key_textbox.change(set_openai_api_key,
                                  inputs=[openai_api_key_textbox],
                                  outputs=[chain_state, llm_state])

block.launch(debug=True)

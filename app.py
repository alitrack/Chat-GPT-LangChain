import io
import os
from typing import Optional, Tuple
import datetime
import gradio as gr
import requests

# UNCOMMENT TO USE WHISPER
# import warnings
# import whisper

from langchain import ConversationChain, LLMChain

from langchain.agents import load_tools, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import OpenAI

# Console to variable
from io import StringIO
import sys
import re

from openai.error import AuthenticationError, InvalidRequestError

# Pertains to Express-inator functionality
from langchain.prompts import PromptTemplate

news_api_key = os.environ["NEWS_API_KEY"]
tmdb_bearer_token = os.environ["TMDB_BEARER_TOKEN"]

TOOLS_LIST = ['serpapi', 'wolfram-alpha', 'google-search', 'pal-math', 'pal-colored-objects', 'news-api', 'tmdb-api',
              'open-meteo-api']
TOOLS_DEFAULT_LIST = ['serpapi', 'pal-math']
BUG_FOUND_MSG = "Congratulations, you've found a bug in this application!"
AUTH_ERR_MSG = "Please paste your OpenAI key."

# Pertains to Express-inator functionality
NUM_WORDS_DEFAULT = 0
FORMALITY_DEFAULT = "N/A"
TEMPERATURE_DEFAULT = 0.5
EMOTION_DEFAULT = "N/A"
TRANSLATE_TO_DEFAULT = "N/A"
LITERARY_STYLE_DEFAULT = "N/A"
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["original_words", "num_words", "formality", "emotions", "translate_to", "literary_style"],
    template="Restate {num_words}{formality}{emotions}{translate_to}{literary_style}the following: \n{original_words}\n",
)


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


# Pertains to Express-inator functionality
def transform_text(desc, express_chain, num_words, formality,
                   anticipation_level, joy_level, trust_level,
                   fear_level, surprise_level, sadness_level, disgust_level, anger_level,
                   translate_to, literary_style):
    num_words_prompt = ""
    if num_words and int(num_words) != 0:
        num_words_prompt = "using up to " + str(num_words) + " words, "

    # Change some arguments to lower case
    formality = formality.lower()
    anticipation_level = anticipation_level.lower()
    joy_level = joy_level.lower()
    trust_level = trust_level.lower()
    fear_level = fear_level.lower()
    surprise_level = surprise_level.lower()
    sadness_level = sadness_level.lower()
    disgust_level = disgust_level.lower()
    anger_level = anger_level.lower()

    formality_str = ""
    if formality != "n/a":
        formality_str = "in a " + formality + " manner, "

    # put all emotions into a list
    emotions = []
    if anticipation_level != "n/a":
        emotions.append(anticipation_level)
    if joy_level != "n/a":
        emotions.append(joy_level)
    if trust_level != "n/a":
        emotions.append(trust_level)
    if fear_level != "n/a":
        emotions.append(fear_level)
    if surprise_level != "n/a":
        emotions.append(surprise_level)
    if sadness_level != "n/a":
        emotions.append(sadness_level)
    if disgust_level != "n/a":
        emotions.append(disgust_level)
    if anger_level != "n/a":
        emotions.append(anger_level)

    emotions_str = ""
    if len(emotions) > 0:
        if len(emotions) == 1:
            emotions_str = "with emotion of " + emotions[0] + ", "
        else:
            emotions_str = "with emotions of " + ", ".join(emotions[:-1]) + " and " + emotions[-1] + ", "

    translate_to_str = ""
    if translate_to != TRANSLATE_TO_DEFAULT:
        translate_to_str = "translated to " + translate_to + ", "

    literary_style_str = ""
    if literary_style != LITERARY_STYLE_DEFAULT:
        if literary_style == "Prose":
            literary_style_str = "as prose, "
        elif literary_style == "Poetry":
            literary_style_str = "as a poem, "
        elif literary_style == "Haiku":
            literary_style_str = "as a haiku, "
        elif literary_style == "Limerick":
            literary_style_str = "as a limerick, "
        elif literary_style == "Joke":
            literary_style_str = "as a very funny joke with a setup and punchline, "
        elif literary_style == "Knock-knock":
            literary_style_str = "as a very funny knock-knock joke, "

    formatted_prompt = PROMPT_TEMPLATE.format(
        original_words=desc,
        num_words=num_words_prompt,
        formality=formality_str,
        emotions=emotions_str,
        translate_to=translate_to_str,
        literary_style=literary_style_str
    )

    trans_instr = num_words_prompt + formality_str + emotions_str + translate_to_str + literary_style_str
    if express_chain and len(trans_instr.strip()) > 0:
        generated_text = express_chain.run(
            {'original_words': desc, 'num_words': num_words_prompt, 'formality': formality_str,
             'emotions': emotions_str, 'translate_to': translate_to_str,
             'literary_style': literary_style_str}).strip()
    else:
        print("Not transforming text")
        generated_text = desc

    # replace all newlines with <br> in generated_text
    generated_text = generated_text.replace("\n", "<br>")

    prompt_plus_generated = "<b>GPT prompt:</b> " + formatted_prompt + "<br/><br/><code>" + generated_text + "</code>"

    print("\n==== date/time: " + str(datetime.datetime.now() - datetime.timedelta(hours=5)) + " ====")
    print("prompt_plus_generated: " + prompt_plus_generated)

    return generated_text


def load_chain(tools_list, llm):
    chain = None
    express_chain = None
    if llm:
        print("tools_list", tools_list)
        tool_names = tools_list
        tools = load_tools(tool_names, llm=llm, news_api_key=news_api_key, tmdb_bearer_token=tmdb_bearer_token)

        memory = ConversationBufferMemory(memory_key="chat_history")

        chain = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=memory)
        express_chain = LLMChain(llm=llm, prompt=PROMPT_TEMPLATE, verbose=True)

    return chain, express_chain


def set_openai_api_key(api_key):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key:
        llm = OpenAI(temperature=0, openai_api_key=api_key)
        chain, express_chain = load_chain(TOOLS_DEFAULT_LIST, llm)
        return chain, express_chain, llm


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
        trace_chain: bool, express_chain: Optional[LLMChain], num_words, formality, anticipation_level, joy_level,
        trust_level,
        fear_level, surprise_level, sadness_level, disgust_level, anger_level,
        translate_to, literary_style):
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

    output = transform_text(output, express_chain, num_words, formality, anticipation_level, joy_level, trust_level,
                            fear_level, surprise_level, sadness_level, disgust_level, anger_level,
                            translate_to, literary_style)

    text_to_display = output
    if trace_chain:
        text_to_display = hidden_text + "\n\n" + output
    history.append((inp, text_to_display))

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
        chain, express_chain = load_chain(state, llm)
        return state, llm, chain, express_chain


def update_foo(widget, state):
    if widget:
        state = widget
        return state


with gr.Blocks(css=".gradio-container {background-color: lightgray max-height: 1000} ") as block:
    llm_state = gr.State()
    history_state = gr.State()
    chain_state = gr.State()
    express_chain_state = gr.State()
    tools_list_state = gr.State(TOOLS_DEFAULT_LIST)
    trace_chain_state = gr.State(False)

    # Pertains to Express-inator functionality
    num_words_state = gr.State(NUM_WORDS_DEFAULT)
    formality_state = gr.State(FORMALITY_DEFAULT)
    anticipation_level_state = gr.State(EMOTION_DEFAULT)
    joy_level_state = gr.State(EMOTION_DEFAULT)
    trust_level_state = gr.State(EMOTION_DEFAULT)
    fear_level_state = gr.State(EMOTION_DEFAULT)
    surprise_level_state = gr.State(EMOTION_DEFAULT)
    sadness_level_state = gr.State(EMOTION_DEFAULT)
    disgust_level_state = gr.State(EMOTION_DEFAULT)
    anger_level_state = gr.State(EMOTION_DEFAULT)
    translate_to_state = gr.State(TRANSLATE_TO_DEFAULT)
    literary_style_state = gr.State(LITERARY_STYLE_DEFAULT)

    with gr.Row():
        with gr.Column():
            gr.Markdown("<h4><center>Conversational Agent using GPT-3.5 & LangChain</center></h4>")

        openai_api_key_textbox = gr.Textbox(placeholder="Paste your OpenAI API key (sk-...)",
                                            show_label=False, lines=1, type='password')

    with gr.Row():
        with gr.Column(scale=1, min_width=240):
            my_file = gr.File(label="Upload a file", type="file", visible=False)
            tmp_file = gr.File("videos/Masahiro.mp4", visible=False)
            tmp_file_url = "/file=" + tmp_file.value['name']
            htm_video = f'<video width="256" height="256" autoplay muted loop><source src={tmp_file_url} type="video/mp4" poster="Masahiro.png"></video>'
            video_html = gr.HTML(htm_video)

            trace_chain_cb = gr.Checkbox(label="Show chain", value=False)
            trace_chain_cb.change(update_foo, inputs=[trace_chain_cb, trace_chain_state],
                                  outputs=[trace_chain_state])

        with gr.Column(scale=3):
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
        with gr.Column():
            with gr.Accordion("Tools", open=False):
                tools_cb_group = gr.CheckboxGroup(label="Tools:", choices=TOOLS_LIST,
                                                  value=TOOLS_DEFAULT_LIST)
                tools_cb_group.change(update_selected_tools,
                                      inputs=[tools_cb_group, tools_list_state, llm_state],
                                      outputs=[tools_list_state, llm_state, chain_state, express_chain_state])

            with gr.Accordion("Formality", open=False):
                formality_radio = gr.Radio(label="Formality:",
                                           choices=[FORMALITY_DEFAULT, "Casual", "Polite", "Honorific"],
                                           value=FORMALITY_DEFAULT)
                formality_radio.change(update_foo,
                                       inputs=[formality_radio, formality_state],
                                       outputs=[formality_state])

            with gr.Accordion("Translate to", open=False):
                translate_to_radio = gr.Radio(label="Translate to:", choices=[
                    TRANSLATE_TO_DEFAULT, "Arabic", "British English", "Chinese (Simplified)", "Chinese (Traditional)",
                    "Czech", "Danish", "Dutch", "English", "Finnish", "French", "German",
                    "Greek", "Hebrew", "Hindi", "Hungarian", "Indonesian", "Italian", "Japanese",
                    "Korean", "Norwegian", "Old English", "Polish", "Portuguese", "Romanian",
                    "Russian", "Spanish", "Swedish", "Thai", "Turkish",
                    "Vietnamese",
                    "emojis", "Gen Z slang", "how the stereotypical Karen would say it", "Klingon",
                    "Pirate", "Strange Planet expospeak technical talk", "Yoda"],
                                              value=TRANSLATE_TO_DEFAULT)

                translate_to_radio.change(update_foo,
                                          inputs=[translate_to_radio, translate_to_state],
                                          outputs=[translate_to_state])

        with gr.Column():
            with gr.Accordion("Literary style", open=False):
                literary_style_radio = gr.Radio(label="Literary style:", choices=[
                    LITERARY_STYLE_DEFAULT, "Poetry", "Haiku", "Limerick", "Joke", "Knock-knock"],
                                                value=LITERARY_STYLE_DEFAULT)

                literary_style_radio.change(update_foo,
                                            inputs=[literary_style_radio, literary_style_state],
                                            outputs=[literary_style_state])

            with gr.Accordion("Emotions", open=False):
                anticipation_level_radio = gr.Radio(label="Anticipation level:",
                                                    choices=[EMOTION_DEFAULT, "Interest", "Anticipation", "Vigilance"],
                                                    value=EMOTION_DEFAULT)
                anticipation_level_radio.change(update_foo,
                                                inputs=[anticipation_level_radio, anticipation_level_state],
                                                outputs=[anticipation_level_state])

                joy_level_radio = gr.Radio(label="Joy level:",
                                           choices=[EMOTION_DEFAULT, "Serenity", "Joy", "Ecstasy"],
                                           value=EMOTION_DEFAULT)
                joy_level_radio.change(update_foo,
                                       inputs=[joy_level_radio, joy_level_state],
                                       outputs=[joy_level_state])

                trust_level_radio = gr.Radio(label="Trust level:",
                                             choices=[EMOTION_DEFAULT, "Acceptance", "Trust", "Admiration"],
                                             value=EMOTION_DEFAULT)
                trust_level_radio.change(update_foo,
                                         inputs=[trust_level_radio, trust_level_state],
                                         outputs=[trust_level_state])

                fear_level_radio = gr.Radio(label="Fear level:",
                                            choices=[EMOTION_DEFAULT, "Apprehension", "Fear", "Terror"],
                                            value=EMOTION_DEFAULT)
                fear_level_radio.change(update_foo,
                                        inputs=[fear_level_radio, fear_level_state],
                                        outputs=[fear_level_state])

                surprise_level_radio = gr.Radio(label="Surprise level:",
                                                choices=[EMOTION_DEFAULT, "Distraction", "Surprise", "Amazement"],
                                                value=EMOTION_DEFAULT)
                surprise_level_radio.change(update_foo,
                                            inputs=[surprise_level_radio, surprise_level_state],
                                            outputs=[surprise_level_state])

                sadness_level_radio = gr.Radio(label="Sadness level:",
                                               choices=[EMOTION_DEFAULT, "Pensiveness", "Sadness", "Grief"],
                                               value=EMOTION_DEFAULT)
                sadness_level_radio.change(update_foo,
                                           inputs=[sadness_level_radio, sadness_level_state],
                                           outputs=[sadness_level_state])

                disgust_level_radio = gr.Radio(label="Disgust level:",
                                               choices=[EMOTION_DEFAULT, "Boredom", "Disgust", "Loathing"],
                                               value=EMOTION_DEFAULT)
                disgust_level_radio.change(update_foo,
                                           inputs=[disgust_level_radio, disgust_level_state],
                                           outputs=[disgust_level_state])

                anger_level_radio = gr.Radio(label="Anger level:",
                                             choices=[EMOTION_DEFAULT, "Annoyance", "Anger", "Rage"],
                                             value=EMOTION_DEFAULT)
                anger_level_radio.change(update_foo,
                                         inputs=[anger_level_radio, anger_level_state],
                                         outputs=[anger_level_state])

            with gr.Accordion("Max words", open=False):
                num_words_slider = gr.Slider(label="Max number of words to generate (0 for don't care)",
                                             value=NUM_WORDS_DEFAULT, minimum=0, maximum=100, step=10)
                num_words_slider.change(update_foo,
                                        inputs=[num_words_slider, num_words_state],
                                        outputs=[num_words_state])

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

    message.submit(chat, inputs=[message, history_state, chain_state, trace_chain_state,
                                 express_chain_state, num_words_state, formality_state,
                                 anticipation_level_state, joy_level_state, trust_level_state, fear_level_state,
                                 surprise_level_state, sadness_level_state, disgust_level_state, anger_level_state,
                                 translate_to_state, literary_style_state],
                   outputs=[chatbot, history_state, video_html, my_file, message])

    submit.click(chat, inputs=[message, history_state, chain_state, trace_chain_state,
                               express_chain_state, num_words_state, formality_state,
                               anticipation_level_state, joy_level_state, trust_level_state, fear_level_state,
                               surprise_level_state, sadness_level_state, disgust_level_state, anger_level_state,
                               translate_to_state, literary_style_state],
                 outputs=[chatbot, history_state, video_html, my_file, message])

    openai_api_key_textbox.change(set_openai_api_key,
                                  inputs=[openai_api_key_textbox],
                                  outputs=[chain_state, express_chain_state, llm_state])

block.launch(debug=True)

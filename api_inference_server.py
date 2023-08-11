from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from Conversation.conversation import character_msg_constructor
from Conversation.translation.pipeline import Translate
from AIVoifu.tts import tts # text to speech from huggingface
from vtube_studio import Char_control
import romajitable # temporary use this since It'll blow up our ram if we use Machine Translation Model
import scipy.io.wavfile as wavfile
import torch
import wget 
from pysentimiento import create_analyzer
import re

# ---------- Config ----------
translation = bool(input("Enable translation? (Y/n): ").lower() in ['y', ''])

device = torch.device('cpu') # default to cpu
use_gpu = torch.cuda.is_available()
print("Detecting GPU...")
if use_gpu:
    print("GPU detected!")
    device = torch.device('cuda')
    print("Using GPU? (Y/N)")
    if input().lower() == 'y':
        print("Using GPU...")
    else:
        print("Using CPU...")
        use_gpu = False
        device = torch.device('cpu')

# ---------- load Conversation model ----------
print("Initilizing model....")
print("Loading language model...")
tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-1.3b", use_fast=True)
config = AutoConfig.from_pretrained("PygmalionAI/pygmalion-1.3b", is_decoder=True)
model = AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-1.3b", config=config, )
senti_analyzer = create_analyzer(task="sentiment", lang="en")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
# config = AutoConfig.from_pretrained("bert-base-uncased", is_decoder=True)
# model = AutoModelForCausalLM.from_pretrained("bert-base-uncased", config=config, )

if use_gpu: # load model to GPU
  model = model.to(device)
  print("Inference at half precision? (Y/N)")
  if input().lower() == 'y':
      print("Loading model at half precision...")
      model.half()
  else:
      print("Loading model at full precision...")

if translation:
    print("Translation enabled!")
    print("Loading machine translation model...")
    translator = Translate(device) # initialize translator
else:
    print("Translation disabled!")
    print("Proceeding... wtih pure english conversation")

print('--------Finished!----------')
# --------------------------------------------------

# --------- Define Waifu personality ----------
talk = character_msg_constructor('Lilia', """Species("Elf")
Mind("sexy" + "cute" + "Loving" + "Based as Fuck")
Personality("sexy" + "cute"+ "kind + "Loving" + "Based as Fuck")
Body("160cm tall" + "5 foot 2 inches tall" + "small breasts" + "white" + "slim")
Description("Lilia is 18 years old girl" + "she love pancake")
Loves("Cats" + "Birds" + "Waterfalls")
Sexual Orientation("Straight" + "Hetero" + "Heterosexual")""")
# ---------------------------------------------

### --- websocket server setup
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import json
import asyncio

# use fast api instead
app = FastAPI()


def senti_analyze(text: str) -> list:
    emotions_text = text
    if '*' in text:
        emotions_text = re.findall(r'\*(.*?)\*', emotions_text)  # get emotion *action* as input if exist
        emotions_text = ' '.join(emotions_text)  # create input

    senti = senti_analyzer.predict(emotions_text)
    return senti

opt = torch.optim.SGD(model.parameters(), lr=0.0000001)
loss_fn = torch.nn.L1Loss()
def train_model_using_user_response(model: torch.nn.Module, model_input, user_response):
    senti = senti_analyze(user_response)
    model_output = model(**model_input)
    # print(f"[debug]: {model_output.logits}")
    if senti.output == "POS" and senti.probas["POS"] > 0.9:
        print(f"User Happy")
        pos = torch.zeros_like(model_output.logits)+senti.probas["POS"]
        loss = loss_fn(model_output.logits, pos)
        loss.backward()
        opt.step()
    elif senti.output == "NEG" and senti.probas["NEG"] > 0.9:
        print(f"User not Happy")
        neg = torch.ones_like(model_output.logits)-senti.probas["NEG"]
        loss = loss_fn(model_output.logits, neg)
        loss.backward()
        opt.step()


last_data = None
# do a http server instead
@app.get("/waifuapi")
async def get_waifuapi(command: str, data: str):
    global last_data
    if command == "chat":
        if last_data is not None:
            train_model_using_user_response(model, last_data, data)
        msg = data
        # ----------- Create Response --------------------------
        msg = talk.construct_msg(msg, talk.history_loop_cache) # construct message input and cache History model
        print(f"[debug msg]: {msg}")
        ## ----------- Will move this to server later -------- (16GB ram needed at least)
        inputs = tokenizer(msg, return_tensors='pt')
        if use_gpu:
            inputs = inputs.to(device)
        out = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 100, pad_token_id=tokenizer.eos_token_id)
        last_data = inputs
        conversation = tokenizer.decode(out[0])
        ## --------------------------------------------------

        ## get conversation in proper format and create history from [last_idx: last_idx+2] conversation
        talk.split_counter += 2
        current_converse = talk.get_current_converse(conversation)[:talk.split_counter][talk.split_counter-2:talk.split_counter]
        print(conversation) # only print waifu answer since input already show
        talk.history_loop_cache = '\n'.join(current_converse) # update history for next input message
        print(f"[debug history_cache]: {talk.history_loop_cache}")

        # -------------- use machine translation model to translate to japanese and submit to client --------------
        if len(current_converse) < 1:
            current_converse = ["sorry, internal error"]
        cleaned_text = talk.clean_emotion_action_text_for_speech(current_converse[-1]) # clean text for speech
        
        translated = '' # initialize translated text as empty by default
        if translation:
            translated = translator.translate(cleaned_text) # translate to [language] if translation is enabled

        return JSONResponse(content=f'{current_converse[-1]}<split_token>{translated}')
    
    if command == "reset":
        talk.conversation_history = ''
        talk.history_loop_cache = ''
        talk.split_counter = 0
        return JSONResponse(content='Story reseted...')

if __name__ == "__main__":
    import uvicorn
    import socket # check if port is available
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 8267
    try:
        s.bind(("localhost", port))
        s.close()
    except socket.error as e:
        print(f"Port {port} is already in use")
        exit()
    uvicorn.run(app, host="0.0.0.0", port=port)
